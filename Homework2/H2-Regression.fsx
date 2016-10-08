#load @"..\packages\FSharp.Charting.0.90.14\FSharp.Charting.fsx"
#load @"..\packages\MathNet.Numerics.FSharp.3.13.1\MathNet.Numerics.fsx"

open System
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double
open MathNet.Numerics.Distributions
open MathNet.Numerics.Statistics
open FSharp.Charting


let rand = new Random()
let mutable dataPoints = [for i in 1..10 -> Vector.Build.Random(2, new ContinuousUniform(-1.0,1.0) )]
let mutable linePoints =  [ for i in 1..2 -> Vector.Build.Random(2, new ContinuousUniform(-1.0,1.0) )]
let mutable foundPoints =  [(1.0, 1.0);(1.0,1.0)]
let mutable correctLabels = ["1";"-1"]

let init N =
    dataPoints <- [for i in 1..N -> Vector.Build.Random(2, new ContinuousUniform(-1.0,1.0) )]
    linePoints <- [ for i in 1..2 -> Vector.Build.Random(2, new ContinuousUniform(-1.0,1.0) )]
//    linePoints <- [ Vector.Build.Dense([|-0.40877; -0.653459|]);
//                           Vector.Build.Dense([|-0.579892;  -0.84767|])];
//
//
//    dataPoints <- [
//                           Vector.Build.Dense([|0.308796; -0.758407|]);
//                           Vector.Build.Dense([|-0.173793; 0.120863|]);
//                           Vector.Build.Dense([|0.841236; -0.780489|]);
//                           Vector.Build.Dense([|-0.291265; 0.129208|]);
//                           Vector.Build.Dense([|-0.812164; -0.697282|]); ]


let label_point(line: Vector<float>, point: Vector<float>) =
    let point3 = Vector.Build.Dense([|1.0;point.[0];point.[1]|])
    let mult = point3*line
    if  mult > 0.0
    then (point,1.0)
    else (if mult = 0.0 then (point,0.0) else (point,-1.0))

let find_line(point1: Vector<float>, point2: Vector<float>) =
    let a = Matrix<float>.Build.DenseOfRowArrays( [| point1.ToArray(); point2.ToArray() |])
    let w0 = 1.0
    let b = vector [-w0; -w0]
    let result = a.Solve(b).ToArray()
    vector [w0; result.[0]; result.[1]]

let regression correct_labels =
    let normalizedPoints = dataPoints |> List.map (fun v -> [|1.0; v.[0]; v.[1]|])
    let x = Matrix.Build.DenseOfRowArrays(normalizedPoints)
    let y = Vector.Build.DenseOfEnumerable(correct_labels).ToColumnMatrix()
    let xT = x.Transpose()
    let w = ((xT*x).Inverse())*xT*y
    vector [w.[0,0]; w.[1,0]; w.[2,0]]

let calculateP(f, g) =
    let points = dataPoints
    let label_f = points |> List.map (fun v -> label_point(f, v))
    let label_g = points |> List.map (fun v -> label_point(g, v))

    correctLabels <- label_f |> List.map (fun (f, fs) -> fs.ToString())

    let difference = List.zip label_f label_g |> List.map (fun((f,fs),(g,gs)) -> if fs <> gs then 1.0 else 0.0)
    Statistics.Mean difference

let get2Points (w: Vector<float>) =
    let x = [0.5; -0.5]
    let y = List.map (fun x -> -1.0*(w.[0] + w.[1]*x)/w.[2]) x
    List.zip x y

let run() =
    init 100


    let f = find_line(linePoints.[0], linePoints.[1])
    let correct_labels = dataPoints
                         |> List.map (fun v -> label_point(f, v))
                         |> List.map (fun (x,s) -> s)
    let g = regression correct_labels

    foundPoints <- get2Points(g)
    let p = calculateP(f, g)
    p


printfn "!!! %A" (run())
Chart.Combine(
        [ Chart.Line(foundPoints, Color = Drawing.Color.Red)
          Chart.Line(linePoints |> List.map (fun v -> (v.[0], v.[1])), Color = Drawing.Color.Blue)
          Chart.Point(data = (dataPoints |> List.map (fun v -> (v.[0], v.[1]))), Labels = correctLabels)])