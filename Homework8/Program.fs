open Accord.Math
open Accord.Math.Optimization
open System.IO
open System
open Accord.IO
open Accord.MachineLearning.VectorMachines
open Accord.MachineLearning.VectorMachines.Learning
open LibSVMsharp.Helpers
open LibSVMsharp


let loadData (fileName :string) =
    let values = seq {
            use sr = new StreamReader(fileName)
            while not sr.EndOfStream do
                yield sr.ReadLine()
        }
    values
                 |> Seq.map (fun line -> line.Split([|' '|], StringSplitOptions.RemoveEmptyEntries))
                 |> Seq.map (fun arr -> (((float)arr.[1],(float)arr.[2]), (float)arr.[0]))
                 |> Seq.toArray

let one_vs_one data (number1: float) (number2: float) =
    data |> Array.filter (fun (x,y) -> y = number1 || y = number2)
         |> Array.map (fun (x,y) -> if y = number1 then (x, 1.0) else (x,-1.0))

let one_vs_all data (number: float) =
    data |> Array.map (fun (x,y) -> if y = number then (x, 1.0) else (x,-1.0))

let solveSVM data testData C =
    let problem = new SVMProblem()
    problem.X.AddRange(data |> Array.map (fun ((x1,x2),y) -> [| SVMNode(1,x1); SVMNode(2,x2) |]))
    problem.Y.AddRange(data |> Array.map (fun ((x1,x2),y) -> y))


    let parameter = new SVMParameter();
    parameter.Type <- SVMType.C_SVC;
    parameter.Kernel <- SVMKernelType.POLY;
    parameter.Degree <- 5
    parameter.C <- C;

    let model = SVM.Train(problem, parameter);
    let testResults = testData |> Array.map (fun ((x1,x2),y) -> SVM.Predict(model, [| SVMNode(1,x1); SVMNode(2,x2) |]))
    let testLabels = testData |> Array.map (fun ((x1,x2),y) -> y)
    let ein = Array.sum (Array.zip testResults testLabels |> Array.map (fun (res,lab) -> if res = lab then 0.0 else 1.0)) / (float)testLabels.Length
    (ein, model.TotalSVCount)

[<EntryPoint>]
let main argv =

    let data = loadData "../../features.train"
    let testData = loadData "../../features.test"

    //let eins =  [for i in [0.0..1.0..2.0] -> solveSVM (one_vs_all data i) (one_vs_all testData i) 0.01]
    let eins =  [for i in [0.0001;0.001;0.01;1.0] -> solveSVM (one_vs_one data 1.0 5.0) (one_vs_one data 1.0 5.0) i]

    //let accuracy = SVMHelper.EvaluateClassificationProblem(problem, testResults);

    0



