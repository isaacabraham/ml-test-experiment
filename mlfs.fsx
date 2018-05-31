[<AutoOpen>]
module Microsoft.ML.Fs

type Classifier =
    | BinaryTreeClassifier of leaves:int * trees:int * documentsPerLeaf : int
type Pipeline = obj
type TrainedModel = obj
let createPipeline() = ()
/// Load data into the ML pipeline.
let loadWith<'T> (rows:'T seq) (features:('T -> string) list) (pipeline:Pipeline) = ()
/// Load feature data into the ML pipeline.
let loadWithSimple (featureRows:_ list) (pipeline:Pipeline) = ()


/// Specifies a classifer for the pipeline.
let withClassifier (classifier:Classifier) = id
/// Trains the model.
let train (p:Pipeline) : TrainedModel = obj()
/// Predicts values for the supplied data using a trained model.
let predict<'T, 'Q> (unpredictedData:'T seq) (getValue:'T -> 'Q) (model:TrainedModel) =
    unpredictedData |> Seq.map(fun x -> x, true)

let predictSimple (unpredictedData:_ seq) (model:TrainedModel) =
    unpredictedData |> Seq.map(fun x -> x, true)    