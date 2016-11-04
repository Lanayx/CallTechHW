open System.IO
open System
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.Statistics

let regression (dataPoints: (float*float) list) correct_labels  =
    let normalizedPoints = dataPoints |>
                           List.map (fun (x1,x2) -> [|1.0; x1; x2; x1*x1; x2*x2; x1*x2; abs(x1-x2); abs(x1+x2) |])
    let x = Matrix.Build.DenseOfRowArrays(normalizedPoints)
    let y = Vector.Build.DenseOfEnumerable(correct_labels)
    let xT = x.Transpose()
    let w = (xT*x).Inverse()*xT*y
    w

let readData (fileName: string) =
    let values = seq {
            use sr = new StreamReader(fileName)
            while not sr.EndOfStream do
                yield sr.ReadLine()
        }
    values
         |> Seq.map (fun line -> line.Split([|' '|], StringSplitOptions.RemoveEmptyEntries))
         |> Seq.map (fun arr -> (((float)arr.[0],(float)arr.[1]), (float)arr.[2]))
         |> Seq.toList

let label_point(line: Vector<float>, point: (float*float)) =
    let (x1,x2) = point
    let point3 = vector [1.0; x1; x2; x1*x1; x2*x2; x1*x2; abs(x1-x2); abs(x1+x2)]
    let mult = point3*line
    if  mult > 0.0
    then (point,1.0)
    else (if mult = 0.0 then (point,0.0) else (point,-1.0))

let calculateP (dataPoints: (float*float) list) correct_labels g =
    let points = dataPoints
    let label_g = points |> List.map (fun v -> label_point(g, v))
    let difference = List.zip correct_labels label_g |> List.map (fun(fs,(g,gs)) -> if fs <> gs then 1.0 else 0.0)
    Statistics.Mean difference

//let einCalc (dataPoints: (float*float) list) (correct_labels: float list) (w: Vector<float>) =
//    let normalizedPoints = dataPoints |>
//                           List.map (fun (x1,x2) -> [|1.0; x1; x2; x1*x1; x2*x2; x1*x2; abs(x1-x2); abs(x1+x2) |])
//    let x = Matrix.Build.DenseOfRowArrays(normalizedPoints)
//    let y = Vector.Build.DenseOfEnumerable(correct_labels)
//    let xwAdjusted = (x*w).Map(fun res -> match res with
//                                            | a when a > 0.0 -> 1.0
//                                            | b when b = 0.0 -> 0.0
//                                            | _ -> -1.0)
//    let ein = 1.0/((float)dataPoints.Length)*(((xwAdjusted - y).L2Norm())**2.0)
//    ein

[<EntryPoint>]
let main argv =
    let testData = readData "../../in.dta"
    let verificationData = readData "../../out.dta"
    let testX = testData |> List.map (fun (x,y) -> x)
    let testY = testData |> List.map (fun (x,y) -> y)
    let verifyX = verificationData |> List.map (fun (x,y) -> x)
    let verifyY = verificationData |> List.map (fun (x,y) -> y)
    let w = regression testX testY
    let inP = calculateP testX testY w
    let outP = calculateP verifyX verifyY w
    printfn "%A %A" inP outP
    0 // return an integer exit code
