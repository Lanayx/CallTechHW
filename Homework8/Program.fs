open Accord.Math
open Accord.Math.Optimization
open System.IO
open System
open Accord.IO
open Accord.MachineLearning.VectorMachines
open Accord.MachineLearning.VectorMachines.Learning
open LibSVMsharp.Helpers
open LibSVMsharp

let rand = new Random()

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
    parameter.Degree <- 2
    parameter.C <- C;
    //parameter.Gamma <- 1.0;

    let model = SVM.Train(problem, parameter);
    let testResults = testData |> Array.map (fun ((x1,x2),y) -> SVM.Predict(model, [| SVMNode(1,x1); SVMNode(2,x2) |]))
    let testLabels = testData |> Array.map (fun ((x1,x2),y) -> y)
    let ein = Array.sum (Array.zip testResults testLabels |> Array.map (fun (res,lab) -> if res = lab then 0.0 else 1.0)) / (float)testLabels.Length
    (ein, model.TotalSVCount)

let swap (a: _[]) x y =
    let tmp = a.[x]
    a.[x] <- a.[y]
    a.[y] <- tmp

// shuffle an array (in-place)
let shuffle a =
    Array.iteri (fun i _ -> swap a i (rand.Next(i, Array.length a))) a

let divideForXValidation data =
    let shuffled = List.toArray( Array.toList data)
    shuffle shuffled
    Array.chunkBySize (data.Length/10) shuffled

[<EntryPoint>]
let main argv =
    //Q=2 0.0044843049327354259/34 Q=5 0.0051249199231262/27


    let data = loadData "../../features.train"
    let testData = loadData "../../features.test"

    //let eins =  [for i in [0.0..1.0..2.0] -> solveSVM (one_vs_all data i) (one_vs_all testData i) 0.01]
    //let eins =  [for i in [0.0001;0.001;0.01;1.0] -> solveSVM (one_vs_one data 1.0 5.0) (one_vs_one testData 1.0 5.0) i]

    let initialData = one_vs_one data 1.0 5.0
    let steps = [1..100] |> List.map (fun step ->
        let finalResult = [for c in [0.0001;0.001;0.01;0.1;1.0] ->
                            let chunks = divideForXValidation initialData
                            let result = [0..9] |> List.map (fun i ->
                                                let testData = chunks.[i]
                                                let workData = Array.concat  [| for j in 0..9 -> if i=j then [||] else chunks.[j] |]
                                                let (err, svmCount) = solveSVM (Array.concat [| workData; chunks.[10] |]) testData c
                                                err)
                            let ecv = (List.sum result) / 10.0
                            (ecv, c)]
        let x = finalResult |> List.minBy (fun (ecv, c) -> ecv)
        x )
    let final = steps |> List.groupBy (fun (ecv,c) -> c) |> List.map (fun (key, group) -> (key, group.Length, (List.average (List.map (fun (ecv,c) -> ecv) group))))
    printf "%A" final

    //let eins =  [for i in [0.001;1.0;100.0;10000.0;1000000.0] -> solveSVM (one_vs_one data 1.0 5.0) (one_vs_one testData 1.0 5.0) i]

    0



