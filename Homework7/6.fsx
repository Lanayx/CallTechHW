#load @"..\packages\FSharp.Charting.0.90.14\FSharp.Charting.fsx"
#load @"..\packages\MathNet.Numerics.FSharp.3.14.0-beta01\MathNet.Numerics.fsx"

open System
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double
open MathNet.Numerics.Distributions
open MathNet.Numerics.Statistics
open FSharp.Charting


let src = Vector.Build.Random(20000, new ContinuousUniform(0.0,1.0) ).ToArray()
let e1 = Array.skip 10000 src
let e2 = Array.take 10000 src
let e = Array.zip e1 e2 |>
        Array.map (fun (e1,e2) -> min e1 e2)

let meanE1 = Array.average e1
let meanE2 = Array.average e2
let meanE = Array.average e