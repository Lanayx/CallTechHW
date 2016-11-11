#load @"..\packages\FSharp.Charting.0.90.14\FSharp.Charting.fsx"
#load @"..\packages\MathNet.Numerics.FSharp.3.14.0-beta01\MathNet.Numerics.fsx"

open System
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double
open MathNet.Numerics.Distributions
open MathNet.Numerics.Statistics
open FSharp.Charting


let guesses = [ sqrt(sqrt(3.0)+4.0); sqrt(sqrt(3.0)-1.0);  sqrt(9.0+4.0*sqrt(6.0)); sqrt(9.0-sqrt(6.0)) ]
let sets = guesses |> List.map (fun guess -> [
                                                [ (-1.0, 0.0); (1.0, 0.0) ]
                                                [ (-1.0, 0.0); (guess, 1.0); ]
                                                [ (guess, 1.0); (1.0, 0.0) ]
                                             ])
let pointsSet =  guesses |> List.map (fun guess ->  [ (-1.0, 0.0); (guess, 1.0); (1.0, 0.0) ] )



let regression (dataPoints: (float*float) list) k  =
    let normalizedPoints = dataPoints |>
                           List.map (fun (x,y) -> Array.take k [|1.0; x;|])
    let x = Matrix.Build.DenseOfRowArrays(normalizedPoints)
    let y = Vector.Build.DenseOfEnumerable(List.map (fun (x,y) -> y) dataPoints)
    let xT = x.Transpose()
    let w = (xT*x).Inverse()*xT*y
    w

let calcELin (w0, w1) (points: (float*float) list) =
   points |> List.map (fun (x, y) -> (y - w0 - x*w1)**2.0) |> List.sum

let calcEConst (w0) (points: (float*float) list) =
   points |> List.map (fun (x, y) -> (y - w0)**2.0) |> List.sum



let wsConst = sets |> List.map (fun set -> set |>  List.map (fun subset -> regression subset 1) )
let wsLin = sets |> List.map (fun set -> set |>  List.map (fun subset -> regression subset 2) )

let result =
    List.zip wsConst pointsSet |>
        List.map (fun (ws, points) -> ws |> List.map (fun w -> calcEConst w.[0] points) |> List.average )

let result1 =
    List.zip wsLin pointsSet |>
        List.map (fun (ws, points) -> ws |> List.map (fun w -> calcELin (w.[0], w.[1]) points) |> List.average )