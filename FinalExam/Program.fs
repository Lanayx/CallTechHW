// Learn more about F# at http://fsharp.org
// See the 'F# Tutorial' project for more help.
open RegularizedRegression
open SVM
open RBF

[<EntryPoint>]
let main argv =
    //RRRun()
    //SVMRun()
    RBFRun()
    0 // return an integer exit code
