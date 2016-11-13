//#r @"..\packages\Accord.Math.3.3.1-alpha\lib\net45\Accord.Math.Core.dll"
//#r @"..\packages\Accord.3.3.1-alpha\lib\net45\Accord.dll"
//#r @"..\packages\Accord.Math.3.3.1-alpha\lib\net45\Accord.Math.dll"

open Accord.Math
open Accord.Math.Optimization

let find_line(point1: float[], point2: float[]) =
    let a = Matrix.Create(point1,point2)
    let w0 = 1.0
    let b = [|-w0; -w0|]
    let result = Matrix.Solve(a,b);
    [| w0; result.[0]; result.[1] |]

let label_point(line: float[], point: float[]) =
    let point3 = [|1.0;point.[0];point.[1]|]
    let mult = Matrix.Dot(point3,line)
    if  mult > 0.0
    then (point,1.0)
    else (if mult = 0.0 then (point,0.0) else (point,-1.0))

let rec learn dataPoints (misclasified: (float[]*float) list) (w: float[]) correctLabels =
        match misclasified with
            | [] -> w
            | (x,s)::t ->
                let newW =  w.Add([|1.0;x.[0];x.[1]|].Multiply(s))
                let classifiedNew =  dataPoints |> Array.map (fun v -> label_point(newW, v))
                let msklNew = Array.zip correctLabels classifiedNew
                               |> Array.filter (fun(y,(x,s)) -> y <> s)
                               |> Array.map (fun (y, (x, s)) -> (x,y))
                msklNew.Shuffle()
                learn dataPoints (Array.toList msklNew) newW correctLabels

let perceptron_learning dataPoints correctLabels =
    let wInit = Vector.Create(3,0.0)
    let classified =  dataPoints |> Array.map (fun v -> label_point(wInit, v))
    let mskl = Array.zip correctLabels classified
                              |> Array.filter (fun(y,(x,s)) -> y <> s)
                              |> Array.map (fun (y, (x, s)) -> (x,y))
    mskl.Shuffle()
    let w = learn dataPoints (Array.toList mskl) wInit correctLabels
    w.Divide(w.[0])

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

let calculateP(f, g) =
    let points = [| for i in 1..400 -> Vector.Random(2, -1.0, 1.0) |]
    let label_f = points |> Array.map (fun v -> label_point(f, v))
    let label_g = points |> Array.map (fun v -> label_point(g, v))
    let difference = Array.zip label_f label_g |> Array.map (fun((f,fs),(g,gs)) -> if fs <> gs then 1.0 else 0.0)
    Array.average difference

[<EntryPoint>]
let main argv =
    let mutable counter = 0.0
    let mutable comparasions = 0.0
    for i in 1..1000 do
        let N = 10
        let linePoints = [| for i in 1..2 -> Vector.Random(2, -1.0, 1.0) |]
        let dataPoints = [| for i in 1..N -> Vector.Random(2, -1.0, 1.0) |]
        let f = find_line(linePoints.[0], linePoints.[1])
        let points_and_correct_labels = dataPoints |> Array.map (fun v -> label_point(f, v))
        let g1 = perceptron_learning dataPoints (Array.map (fun (point, s) -> s) points_and_correct_labels)
        let g2 = svm_learning points_and_correct_labels

        let p1 = calculateP(f,g1)
        let p2 = calculateP(f,g2)
        if (p1 <> p2) then comparasions <- comparasions + 1.0
        if p2 <= p1 then counter <- counter + 1.0
    printfn "%f" (counter / comparasions)
    0



