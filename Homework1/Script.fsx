#load @"..\packages\FSharp.Charting.0.90.14\FSharp.Charting.fsx"
#load @"..\packages\MathNet.Numerics.FSharp.3.13.1\MathNet.Numerics.fsx"

open System
open FSharp.Charting
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double
open MathNet.Numerics.Distributions


let N = 100
let linePoints = [for i in 1..2 -> Vector.Build.Random(2, new ContinuousUniform(-1.0,1.0) )]
let dataPoints = [for i in 1..N -> (Vector.Build.Random(2, new ContinuousUniform(-1.0,1.0)), i)]

//let linePoints = [ Vector.Build.Dense([|-0.40877; -0.653459|]);
//                   Vector.Build.Dense([|-0.579892;  -0.84767|])];
//
//
//let dataPoints = [ (Vector.Build.Dense([|-0.173793; 0.120863|]), 1);
//                   (Vector.Build.Dense([|0.308796; -0.758407|]),2);
//                   (Vector.Build.Dense([|0.841236; -0.780489|]),3);
//                   (Vector.Build.Dense([|-0.291265; 0.129208|]),4);
//                   (Vector.Build.Dense([|-0.812164; -0.697282|]),5); ]


let isLeft (a: Vector<float>, b: Vector<float>, c: Vector<float>) =
     if ((b.[0] - a.[0])*(c.[1] - a.[1]) - (b.[1] - a.[1])*(c.[0] - a.[0])) > 0.0 then 1.0 else -1.0


let pointsWithSigns = [for (v, i) in dataPoints -> (v, i, isLeft (linePoints.Head,linePoints.Tail.Head, v))]
let y =  pointsWithSigns |> List.map (fun (v, i, s) -> s)

let w = Vector.Build.Dense(2,0.0)
let misclasifiedStart = pointsWithSigns |> List.map (fun (v, i, s) -> (v, i, -1.0))
let mutable counter = 0

let rec learn misclasified w =
    match misclasified with
        | [] -> true
        | (x,i,s)::t ->
            counter <- counter + 1
            let newW = w + x * s
            let result = [ for (x, i) in dataPoints ->  if newW*x > 0.0 then (x,i,1.0) else (x,i,-1.0) ]
            let mskl = List.zip y result
                          |> List.filter (fun(y1,(x,i,s)) -> y1 <> s)
                          |> List.map (fun (y2, (x,i,s)) -> (x,i,y2))
            learn mskl newW

learn misclasifiedStart w

counter

Chart.Combine(
    [ Chart.Line(linePoints |> List.map (fun (v) -> (v.[0], v.[1])))
      Chart.Point(data = (dataPoints |> List.map (fun (v,i) -> (v.[0], v.[1]))),
                    Labels = (pointsWithSigns |> List.map (fun (v,i, sign) -> String.Format("{0}({1})",i, sign))))])
