module RBF

open System
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.Distributions
open MathNet.Numerics.Statistics
open LibSVMsharp
open System.IO

let func x1 x2 =
    (float) (sign (x2-x1+0.25*sin(Math.PI*x1)))

let getData() =
    let points = [ for i in 1..100 -> Vector.Build.Random(2, new ContinuousUniform(-1.0,1.0)) ]
    points |> List.map (fun point -> ((point.[0], point.[1]), func point.[0] point.[1]))

let getTestData() =
    let values = seq {
            use sr = new StreamReader("../../out.dta")
            while not sr.EndOfStream do
                yield sr.ReadLine()
        }
    values
         |> Seq.map (fun line -> line.Split([|' '|], StringSplitOptions.RemoveEmptyEntries))
         |> Seq.map (fun arr -> (((float)arr.[0],(float)arr.[1]), func ((float)arr.[0]) ((float)arr.[1])))
         |> Seq.toList

let rec getNewLLoydClusters (oldCenters: Vector<float> list) (unsupData: Vector<float>[])  =
    let clusters = [ for x in unsupData ->
                        let results = oldCenters |> List.map (fun center -> (center, (x-center).L2Norm()))
                        let (center, norm) = results |> List.minBy (fun (center, norm) -> norm)
                        (x,center) ]
                        |> List.groupBy (fun (x,center) -> center)
                        |> List.map (fun (center, group)-> group |> List.map (fun (x,center) -> x))
    let centers =
        clusters |> List.map (fun cluster -> (cluster
                                 |> List.fold (fun acc elem -> acc+elem) (Vector.Build.Dense(unsupData.[0].Count,0.0))) / (float)cluster.Length)

    if (oldCenters = centers)
    then centers
    else getNewLLoydClusters centers unsupData

let rec LLoydInner K unsupData =
    let init = [ for i in 1..K -> Vector.Build.Random(2, new ContinuousUniform(-1.0,1.0)) ]
    let centers = getNewLLoydClusters init unsupData |> List.toArray
    if (centers.Length <> init.Length)
    then LLoydInner K unsupData
    else centers


let LLoyd K gamma (data:((float*float)*float)list) (testData:((float*float)*float)list)  =
    let unsupData = data |> List.map (fun ((x1,x2),y) -> vector [x1;x2])
    let dataArray = unsupData |> List.toArray
    let ys = vector (data |> List.map (fun (x,y) -> y))
    let centers = LLoydInner K dataArray
    let fMatrix = Matrix.Build.Dense(data.Length, centers.Length+1, fun i j ->
        if (j = 0) then 1.0 else exp(-gamma*((dataArray.[i]-centers.[j-1]).L2Norm())**2.0))
    let fP = fMatrix.PseudoInverse()
    let ws = fP*ys

    let res = testData |> List.map (fun ((x1,x2),y) ->
                                            Array.zip centers (ws.AsArray() |> Array.skip 1)
                                            |> Array.map (fun(cent,w) -> w*exp(-gamma*((vector [x1;x2] - cent).L2Norm())**2.0))
                                            |> Array.sum)
    let results = res |> List.map (fun x -> sign(x+ws.[0]))
    let labels = testData |> List.map (fun ((x1,x2),y) -> y)
    let ein = List.sum (List.zip results labels |> List.map (fun (res,lab) -> if (float)res = lab then 0.0 else 1.0)) / (float)labels.Length
    ein


let SVM gamma data testData =
    let problem = new SVMProblem()
    problem.X.AddRange(data |> List.map (fun ((x1,x2),y) -> [| SVMNode(1,x1); SVMNode(2,x2) |]))
    problem.Y.AddRange(data |> List.map (fun ((x1,x2),y) -> y))
    let parameter = new SVMParameter();
    parameter.Type <- SVMType.C_SVC;
    parameter.Kernel <- SVMKernelType.RBF;
    parameter.Gamma <- gamma;
    parameter.C <- 10000.0;
    let model = SVM.Train(problem, parameter);
    let results = testData |> List.map (fun ((x1,x2),y) -> SVM.Predict(model, [| SVMNode(1,x1); SVMNode(2,x2) |]))
    let labels = testData |> List.map (fun ((x1,x2),y) -> y)
    let e = List.sum (List.zip results labels |> List.map (fun (res,lab) -> if res = lab then 0.0 else 1.0)) / (float)labels.Length
    e

let RBFRun() =
//    let result = [1..150]
//                |> List.map (fun i ->
//                    let trainData = getData()
//                    let testData = getData()
//                    let svmEin = SVM 1.5 trainData trainData
//                    if (svmEin <> 0.0) then 0.0
//                    else
//                        let svmEout = SVM 1.5 trainData testData
//                        let lloydEout = LLoyd 12 1.5 trainData testData
//                        (float)(sign(svmEout - lloydEout)))
//    let finalResult = result
//                        |> List.filter (fun elem -> elem <> 0.0)
//                        |> List.take 100
//                        |> List.groupBy (fun elem -> elem)
//                        |> List.map (fun (key, group) -> (key, group.Length))

//    let res = [1..1000] |> List.map (fun i ->
//                            let result = [|1.5;2.0|]
//                                         |> Array.map (fun gamma ->
//                                            let trainData = getData()
//                                            let testData = getData()
//                                            let lloydEin = LLoyd 9 gamma trainData trainData
//                                            let lloydEout = LLoyd 9 gamma trainData testData
//                                            (lloydEin, lloydEout))
//
//                            (fst result.[0] > fst result.[1], snd result.[0] > snd result.[1]))
//                         |> List.groupBy (fun elem -> elem)
//                         |> List.map (fun (key, group) -> (key, group.Length))

    let res = [1..1000] |> List.map (fun i ->
                                let trainData = getData()
                                let lloydEin = LLoyd 9 1.5 trainData trainData
                                if lloydEin = 0.0 then 1 else 0)
                       |> List.sum

    printfn "%A" res