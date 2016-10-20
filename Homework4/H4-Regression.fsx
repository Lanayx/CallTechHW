#load @"..\packages\MathNet.Numerics.FSharp.3.13.1\MathNet.Numerics.fsx"

open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double
open MathNet.Numerics.Distributions

//let r = System.Random()
//let x = [for n in 1..10000 -> (r.NextDouble()*2.0-1.0, r.NextDouble()*2.0-1.0)]
//let a (x1,x2) = (x1* sin(System.Math.PI*x1) + x2* sin(System.Math.PI*x2))/(x1*x1+x2*x2)
//let result = x |> List.map a |> List.average

let points1 = Vector.Build.Random(10000, new ContinuousUniform(-1.0,1.0)).ToArray()
let points2 = Vector.Build.Random(10000, new ContinuousUniform(-1.0,1.0)).ToArray()
let points = Array.zip points1 points2

let getResult (x1,x2) =
    let x = Matrix.Build.Dense(2,1, [|x1;x2|])
    let y = Matrix.Build.Dense(2,1, [|sin(System.Math.PI*x1);sin(System.Math.PI*x2)|])
    let xT = x.Transpose()
    let w = ((xT*x).Inverse())*xT*y
    w.[0,0]

let result = points |> Array.map getResult |> Array.average