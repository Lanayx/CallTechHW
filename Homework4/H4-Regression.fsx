#load @"..\packages\MathNet.Numerics.FSharp.3.13.1\MathNet.Numerics.fsx"
#load @"..\packages\FSharp.Charting.0.90.14\FSharp.Charting.fsx"

open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double
open MathNet.Numerics.Distributions
open FSharp.Charting

//let r = System.Random()
//let x = [for n in 1..10000 -> (r.NextDouble()*2.0-1.0, r.NextDouble()*2.0-1.0)]
//let a (x1,x2) = (x1* sin(System.Math.PI*x1) + x2* sin(System.Math.PI*x2))/(x1*x1+x2*x2)
//let result = x |> List.map a |> List.average

let points1 = Vector.Build.Random(1000, new ContinuousUniform(-1.0,1.0)).ToArray()
let points2 = Vector.Build.Random(1000, new ContinuousUniform(-1.0,1.0)).ToArray()
let points = Array.zip points1 points2

let getResult (x1,x2) =
    let x = Matrix.Build.Dense(2,1, [|x1;x2|])
    let y = Matrix.Build.Dense(2,1, [|sin(System.Math.PI*x1);sin(System.Math.PI*x2)|])
    let xT = x.Transpose()
    let w = ((xT*x).Inverse())*xT*y
    w.[0,0]

let regressions = points |> Array.map getResult
let result = regressions |> Array.average
let bias = List.average ([for x in -1.0..0.01..1.0 -> (result*x - sin(System.Math.PI*x))**2.0] )


let variance = [|-1.0..0.01..1.0|] |>
                 Array.map (fun (x) -> List.average [for r in regressions -> (r*x - result*x)**2.0]) |>
                 Array.average

//let (x1,x2) = points.[0]
//Chart.Combine(
//        [ Chart.Line([| (-1.0, result*(-1.0)); (1.0, result * 1.0)|])
//          Chart.Point(data = [| (x1,sin(System.Math.PI*x1)) ;  (x2, sin(System.Math.PI*x2)) |])
//
//          Chart.Line([for x in -1.0..0.01..1.0 ->(x, sin(System.Math.PI*x)) ])])

//let (x1,x2) = x.[0]
//Chart.Combine(
//        [ Chart.Line([| (-1.0, result*(-1.0)); (1.0, result * 1.0)|])
//          Chart.Point(data = [| (x1,sin(System.Math.PI*x1)) ;  (x2, sin(System.Math.PI*x2)) |])
//          Chart.Line([for x in -1.0..0.01..1.0 ->(x, sin(System.Math.PI*x))]) ])