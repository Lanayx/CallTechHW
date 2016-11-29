module RBF

open System
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.Distributions
open MathNet.Numerics.Statistics
open LibSVMsharp

let func x1 x2 =
    (float) (sign (x1-x1+0.25*sin(Math.PI*x1)))

let getData() =
    let points = [ for i in 1..100 -> Vector.Build.Random(2, new ContinuousUniform(-1.0,1.0)) ]
    points |> List.map (fun point -> ((point.[0], point.[1]), func point.[0] point.[1]))

let rec getNewLLoydClusters (clusters: Vector<float> list list) (unsupData: Vector<float>[])=
    let centers =
        clusters |> List.map (fun cluster -> (cluster
                                 |> List.fold (fun acc elem -> acc+elem) (Vector.Build.Dense(unsupData.[0].Count,0.0))) / (float)cluster.Length)
    let newClusters = [ for x in unsupData ->
                        let results = centers |> List.map (fun center -> (center, (x-center).L2Norm()))
                        let (center, norm) = results |> List.minBy (fun (center, norm) -> norm)
                        (x,center) ]
                        |> List.groupBy (fun (x,center) -> center)
                        |> List.map (fun (center, group)-> group |> List.map (fun (x,center) -> x))
    if (clusters = newClusters)
    then centers
    else getNewLLoydClusters newClusters unsupData

let rand = new Random()
let swap (a: _[]) x y =
    let tmp = a.[x]
    a.[x] <- a.[y]
    a.[y] <- tmp
let shuffle a =
    Array.iteri (fun i _ -> swap a i (rand.Next(i, Array.length a))) a

let rec LLoydInner K unsupData =
    shuffle unsupData
    let clusters = List.splitInto K (unsupData |> Array.toList)
    let centers = getNewLLoydClusters clusters unsupData |> List.toArray
    if (centers.Length <> clusters.Length)
    then LLoydInner K unsupData
    else centers


let LLoyd K (data:((float*float)*float)list) =
    let unsupData = data |> List.map (fun ((x1,x2),y) -> vector [1.0;x1;x2])
    let dataArray = unsupData |> List.toArray
    let ys = vector (data |> List.map (fun (x,y) -> y))
    let centers = LLoydInner K dataArray
    let fMatrix = Matrix.Build.Dense(data.Length, centers.Length, fun i j ->
        exp(-1.5*((dataArray.[i]-centers.[j]).L2Norm())**2.0))
    let fP = fMatrix.PseudoInverse()
    let ws = fP*ys

    let results = data |> List.map (fun ((x1,x2),y) ->
                                            Array.zip centers (ws.AsArray())
                                            |> Array.map (fun(cent,w) -> w*exp(-1.5*((vector [1.0;x1;x2] - cent).L2Norm())**2.0))
                                            |> Array.sum
                                            |> sign)
    let labels = data |> List.map (fun ((x1,x2),y) -> y)
    let ein = List.sum (List.zip results labels |> List.map (fun (res,lab) -> if (float)res = lab then 0.0 else 1.0)) / (float)labels.Length
    ein


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
//    let result = [1..99]
//                |> List.map (fun i ->
//                    let testData = getData()
//                    let svmEin = SVM testData
//                    if (svmEin = 0.0) then 0.0 else 1.0)
//                |> List.sum

    let result = LLoyd 9 (getData())

    printf "%A" (result)