open System
open FSharp.Charting
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double
open MathNet.Numerics.Distributions

[<EntryPoint>]
let main argv =
    let N = 100
    let linePoints = [for i in 1..2 -> Vector.Build.Random(2, new ContinuousUniform(-1.0,1.0) )]
    let dataPoints = [for i in 1..N -> (Vector.Build.Random(2, new ContinuousUniform(-1.0,1.0)), i)]

//    let linePoints = [ Vector.Build.Dense([|-0.40877; -0.653459|]);
//                       Vector.Build.Dense([|-0.579892;  -0.84767|])];
//
//
//    let dataPoints = [ (Vector.Build.Dense([|-0.173793; 0.120863|]), 1);
//                       (Vector.Build.Dense([|0.308796; -0.758407|]),2);
//                       (Vector.Build.Dense([|0.841236; -0.780489|]),3);
//                       (Vector.Build.Dense([|-0.291265; 0.129208|]),4);
//                       (Vector.Build.Dense([|-0.812164; -0.697282|]),5); ]

    let isLeft (a: Vector<float>, b: Vector<float>, c: Vector<float>) =
         if ((b.[0] - a.[0])*(c.[1] - a.[1]) - (b.[1] - a.[1])*(c.[0] - a.[0])) > 0.0 then 1.0 else -1.0


    let pointsWithSigns = [for (x, i) in dataPoints -> (x, i, isLeft (linePoints.Head,linePoints.Tail.Head, x))]
    let y =  pointsWithSigns |> List.map (fun (v, i, s) -> s)

    let w = Vector.Build.Dense(2,0.0)
    let misclasifiedStart = pointsWithSigns

    let rec learn misclasified w =
        match misclasified with
            | [] -> true
            | (x,i,s)::t ->
                let newW = w + (x * s)
                printfn "%A" w
                let result = [ for (x, i) in dataPoints ->  if newW*x > 0.0 then (x,i,1.0) else (x,i,-1.0) ]
                let mskl = List.zip y result |> List.filter (fun(y1,(x,i,s)) -> y1 <> s) |> List.map (fun (y2, (x,i,s)) -> (x,i,y2))

                learn mskl newW

    learn misclasifiedStart w |> ignore

    0
