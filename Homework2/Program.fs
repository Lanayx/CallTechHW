module Program

//open Regression
open NonLinear
open MathNet.Numerics.Statistics
open MathNet.Numerics.LinearAlgebra

[<EntryPoint>]
let main argv =
    let results = [for i in [1..1000] -> run()]
//    printfn "p:%f" (Statistics.Mean results)
    let pResults = results |> List.map (fun (g, p, p1) -> p)
    let p1Results = results |> List.map (fun (g, p, p1) -> p1)
    let gResults = results |> List.map (fun (g, p, p1) -> g)

    let zeroVector = Vector.Build.Dense(6,0.0)
    printfn "p:%f \np1:%f \ns:%s" (Statistics.Mean pResults)
                                    (Statistics.Mean p1Results)
                                    (((List.fold (fun acc (v) -> acc+v) zeroVector gResults )/1000.0).ToString())
    0 // return an integer exit code
