// Learn more about F# at http://fsharp.org
// See the 'F# Tutorial' project for more help.
open Gradient

[<EntryPoint>]
let main argv =
    //let iterCount = gradDesc (1.0,1.0) 0.1
    let error = coordDesc (1.0,1.0) 0.1
    printfn "%A" error //iterCount
    0 // return an integer exit code
