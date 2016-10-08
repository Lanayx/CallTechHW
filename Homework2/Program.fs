module Program

open Regression
open MathNet.Numerics.Statistics

[<EntryPoint>]
let main argv =
    let results = [for i in [1..1000] -> run()]
    printfn "num:%f" (Statistics.Mean results)
//    let pResults = results |> List.map (fun (p,p1) -> p)
//    let p1Results = results |> List.map (fun (p,p1) -> p1)
//
//    printfn "p:%f p1:%f" (Statistics.Mean pResults) (Statistics.Mean p1Results)
    0 // return an integer exit code
