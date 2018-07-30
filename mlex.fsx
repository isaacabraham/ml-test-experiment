#r "netstandard"
#load @".paket\load\netstandard2.0\main.group.fsx"
#load "mlfs.fsx"

open FSharp.Data
open Microsoft.ML

// 1. Get our data - could be from any source, in this case from a CSV file. This is completely decoupled from the ML library.
type SentimentData = CsvProvider< const (__SOURCE_DIRECTORY__ + @"\data\imdb_labelled.txt"), Separators="\t", HasHeaders=true, IgnoreErrors = true>
let imdbData = SentimentData.GetSample()
















// We can easily work with multiple data sources and e.g. concatinate them.
let yelpData = SentimentData.Load (__SOURCE_DIRECTORY__ + @"\data\yelp_labelled.txt")
let amazonCells = SentimentData.Load (__SOURCE_DIRECTORY__ + @"\data\amazon_cells_labelled.txt")

let allData =
    [ yield! imdbData.Rows
      yield! yelpData.Rows
      yield! amazonCells.Rows ]








/// 2. Train our model using the in-memory data supplied above.
let trained =
    createPipeline()
    |> loadWith allData [ fun r -> r.Text ]
    |> withClassifier (BinaryTreeClassifier(leaves = 5, trees = 5, documentsPerLeaf = 2))
    |> train

/// Alternatively, we use a simpler api which relies on the caller to "prepare" all features.
let _ =
    let featureData = allData |> Seq.map(fun r -> [ r.Text ]) |> Seq.toList
    
    createPipeline()
    |> loadWithSimple featureData






/// 3. Here is some arbitrary data we want to use to predict.
type UnpredictedRow = { Text : string; Source : string }
let predictions =
    [ { Text = "Contoso's 11 is a wonderful experience"; Source = "Amazon" }
      { Text = "Really bad"; Source = "IMDB" }
      { Text = "Joe versus the Volcano Coffee Company is a great film."; Source = "Amazon" } ]

/// 4. We can predict using a function to "retrieve" the value to use for predicting.
let results =
    trained
    |> predict predictions (fun x -> x.Text)
    |> Seq.toArray






/// 5. Or we can use a "simpler" API which relies on the caller to have "extracted" the value.
let extractedPredictions = predictions |> List.map(fun x -> x.Text)
let otherResults =
    trained
    |> predictSimple extractedPredictions
    |> Seq.toArray