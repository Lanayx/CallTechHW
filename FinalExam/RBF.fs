module RBF

open System
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.Distributions
open LibSVMsharp

let func x1 x2 =
    (float) (sign (x1-x1+0.25*sin(Math.PI*x1)))

let getData() =
    let points = [ for i in 1..100 -> Vector.Build.Random(2, new ContinuousUniform(-1.0,1.0)) ]
    points |> List.map (fun point -> ((point.[0], point.[1]), func point.[0] point.[1]))

let LLoyd() =
    1

let SVM data =
    let problem = new SVMProblem()
    problem.X.AddRange(data |> List.map (fun ((x1,x2),y) -> [| SVMNode(1,x1); SVMNode(2,x2) |]))
    problem.Y.AddRange(data |> List.map (fun ((x1,x2),y) -> y))
    let parameter = new SVMParameter();
    parameter.Type <- SVMType.C_SVC;
    parameter.Kernel <- SVMKernelType.RBF;
    parameter.Gamma <- 1.5;
    parameter.C <- 10000.0;
    let model = SVM.Train(problem, parameter);
    let results = data |> List.map (fun ((x1,x2),y) -> SVM.Predict(model, [| SVMNode(1,x1); SVMNode(2,x2) |]))
    let labels = data |> List.map (fun ((x1,x2),y) -> y)
    let ein = List.sum (List.zip results labels |> List.map (fun (res,lab) -> if res = lab then 0.0 else 1.0)) / (float)labels.Length
    ein

let RBFRun() =
    let result = [1..100]
                |> List.map (fun i ->
                    let testData = getData()
                    let svmEin = SVM testData
                    if (svmEin = 0.0) then 0.0 else 1.0)
                |> List.sum

    printf "%A" (result/100.0)