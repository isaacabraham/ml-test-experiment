#r "netstandard"
#load @".paket\load\netstandard2.0\main.group.fsx"

open Microsoft.ML
open Microsoft.ML.Runtime.Api
open Microsoft.ML.Transforms
open Microsoft.ML.Trainers
open System

let nativeDirectory = @"C:\Users\Isaac\.nuget\packages\microsoft.ml\0.1.0\runtimes\win-x64\native"
Environment.SetEnvironmentVariable("Path", Environment.GetEnvironmentVariable("Path") + ";" + nativeDirectory)

let testDataPath = @"data\imdb_labelled.txt"

type SentimentData() =
    [<Column(ordinal = "0"); DefaultValue>]
    val mutable SentimentText : string
    [<Column(ordinal = "1", name = "Label"); DefaultValue>]
    val mutable Sentiment : float32

type SentimentPrediction() =
    [<ColumnName "PredictedLabel"; DefaultValue>]
    val mutable Sentiment : bool

let pipeline = LearningPipeline()
pipeline.Add(TextLoader<SentimentData>(testDataPath, useHeader = false, separator = "tab"))
pipeline.Add(TextFeaturizer("Features", [| "SentimentText" |]))
pipeline.Add(FastTreeBinaryClassifier(NumLeaves = 5, NumTrees = 5, MinDocumentsInLeafs = 2))

let model = pipeline.Train<SentimentData, SentimentPrediction>()

let predictions =
    [ SentimentData(SentimentText = "Contoso's 11 is a wonderful experience")
      SentimentData(SentimentText = "Sort of ok")
      SentimentData(SentimentText = "Joe versus the Volcano Coffee Company is a great film.") ]
    |> model.Predict

predictions
|> Seq.iter(fun p -> printfn "%b" p.Sentiment)