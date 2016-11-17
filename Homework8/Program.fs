open Accord.Math
open Accord.Math.Optimization
open System.IO
open System

let getAlphas (points : (float[]*float) []) =
    let Q = Array2D.create points.Length points.Length 0.0
    for i in 0..points.Length-1 do
        for j in 0..points.Length-1 do
            let (x, y) = points.[i]
            let (x1,y1) = points.[j]
            Q.[i,j] <- y*y1*(Matrix.Dot(x, x1.Transpose()).[0])
    let positiveDefinFix = Matrix.Diagonal(points.Length, 1e-13)// To satisfy library positive definition constraint
    let QFix = Q.Add(positiveDefinFix)
    let d = [|for i in 1..points.Length -> -1.0|]
    let firstRow = [| points |> Array.map (fun (x, y) -> y) |]
    let identity = Matrix.Identity(points.Length)
    let identityJugged = [| for i in 0..points.Length-1 -> identity.GetRow(i)|]
    let A = Matrix.Create(Array.append firstRow identityJugged)
    let b = Vector.Zeros(points.Length+1)
    let solver = new GoldfarbIdnani(QFix,d,A,b,1)
    let success = solver.Minimize()
    solver.Solution

let svm_learning (points : (float[]*float) []) =
    let alphas = (getAlphas points) |> Array.map (fun e -> if e < 1.0 then 0.0 else e)
    let svms = Array.countBy (fun elem ->
                                if (elem > 1.0) then 0 else 1) alphas

    let w = Array.fold2 (fun (acc: float[]) (x: float[], y: float) (alpha:float) -> acc.Add(x.Multiply(alpha*y))) [| 0.0; 0.0 |] points alphas
    let supportVectorIndex = alphas.ArgMax()

    let (x1,y1) = points.[supportVectorIndex]
    let b = 1.0/y1-(Matrix.Dot(w, x1.Transpose()).[0])
    [| b; w.[0]; w.[1]|].Divide(b)

let loadData (fileName :string) =
    let values = seq {
            use sr = new StreamReader(fileName)
            while not sr.EndOfStream do
                yield sr.ReadLine()
        }
    values
         |> Seq.map (fun line -> line.Split([|' '|], StringSplitOptions.RemoveEmptyEntries))
         |> Seq.map (fun arr -> (((float)arr.[1],(float)arr.[2]), (float)arr.[0]))
         |> Seq.toList

[<EntryPoint>]
let main argv =
    let trainData = loadData "../../features.train"
    0



