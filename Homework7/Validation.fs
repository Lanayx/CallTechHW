module Validation

open System.IO
open System
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.Statistics

let regression (dataPoints: (float*float) list) correct_labels k  =
    let normalizedPoints = dataPoints |>
                           List.map (fun (x1,x2) -> Array.take k [|1.0; x1; x2; x1*x1; x2*x2; x1*x2; abs(x1-x2); abs(x1+x2) |])
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

let label_point(line: Vector<float>, point: (float*float)) k =
    let (x1,x2) = point
    let point3 = vector (List.take k [1.0; x1; x2; x1*x1; x2*x2; x1*x2; abs(x1-x2); abs(x1+x2)])
    let mult = point3*line
    if  mult > 0.0
    then (point,1.0)
    else (if mult = 0.0 then (point,0.0) else (point,-1.0))

let calculateP (dataPoints: (float*float) list) correct_labels g k =
    let points = dataPoints
    let label_g = points |> List.map (fun v -> label_point(g, v) k)
    let difference = List.zip correct_labels label_g |> List.map (fun(fs,(g,gs)) -> if fs <> gs then 1.0 else 0.0)
    Statistics.Mean difference

//[<EntryPoint>]
let main argv =
    let testData = readData "../../in.dta"
    let verificationData = readData "../../out.dta"
    let workingData = List.take 25 testData
    let validationData = List.skip 25 testData
    let testX = workingData |> List.map (fun (x,y) -> x)
    let testY = workingData |> List.map (fun (x,y) -> y)
    let validX = validationData |> List.map (fun (x,y) -> x)
    let validY = validationData |> List.map (fun (x,y) -> y)
    let verifyX = verificationData |> List.map (fun (x,y) -> x)
    let verifyY = verificationData |> List.map (fun (x,y) -> y)
    for k in 4..8 do
        let w = regression testX testY k
        let testP = calculateP testX testY w k
        let valP = calculateP validX validY w k
        let verP = calculateP verifyX verifyY w k
        printfn "testP=%A valP=%A verP=%A  k=%i" testP valP verP (k-1)
         // return an integer exit code
    0
