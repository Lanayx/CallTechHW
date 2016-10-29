// Learn more about F# at http://fsharp.org
// See the 'F# Tutorial' project for more help.
open Gradient
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.Distributions
open MathNet.Numerics.Statistics
open System

let rand = new Random()
let mutable counter = 0
let nu = 0.01

let Ein (values : (Vector<float>*float) list) (w: Vector<float>) =
    Statistics.Mean [ for (xn,yn) in values -> 1.0 + exp(-1.0*yn*(w*xn))]

let label_point(line: Vector<float>, point: Vector<float>) =
    let point3 = Vector.Build.Dense([|1.0;point.[0];point.[1]|])
    let mult = point3*line
    if  mult > 0.0
    then (point3,1.0)
    else (if mult = 0.0 then (point3,0.0) else (point3,-1.0))

let find_line(point1: Vector<float>, point2: Vector<float>) =
    let a = Matrix<float>.Build.DenseOfRowArrays( [| point1.ToArray(); point2.ToArray() |])
    let w0 = 1.0
    let b = vector [-w0; -w0]
    let result = a.Solve(b).ToArray();
    vector [w0; result.[0]; result.[1]]

let rec learnFullGrad (w: Vector<float>) (labeled_points: (Vector<float>*float) list) =
    counter <- counter + 1
    let grad = (List.fold (fun acc x -> acc + x) (Vector.Build.Dense(3,0.0)) [ for (xn,yn) in labeled_points -> yn*xn /( 1.0 + exp(yn*(w*xn)))]) / (float) labeled_points.Length
    let deltaW = nu*grad
    match deltaW with
    | x when x.L2Norm() < 0.001 -> w
    | _ -> learnFullGrad (w+deltaW) labeled_points

let rec learnStatGrad (w: Vector<float>) (labeled_points: (Vector<float>*float) list) =
    if labeled_points.IsEmpty then
       w
    else
        counter <- counter + 1
        let (xn, yn) = labeled_points.Head
        let grad = -yn*xn /( 1.0 + exp(yn*(w*xn)))
        let deltaW = nu*grad
        match deltaW with
        | x when x.L2Norm() < 0.001 -> w
        | _ -> learnStatGrad (w-deltaW) labeled_points.Tail

let calculateP(f, g) =
//    """
//    Calculate the probability P(f(x) != g(x)) using Monte Carlo method.
//    `f` and `g` are 3-element arrays [w0, w1, w2] representing a line:
//    w0 + w1 * x + w2 * y = 0.
//    `points` are used when doing Monte Carlo; if not specified, they are
//    generated randomly.
//    """
    let points = [| for i in 1..400 -> Vector.Build.Random(2, new ContinuousUniform(-1.0,1.0)) |]
    let label_f = points |> Array.map (fun v -> label_point(f, v))
    let label_g = points |> Array.map (fun v -> label_point(g, v))
    let difference = Array.zip label_f label_g |> Array.map (fun((f,fs),(g,gs)) -> if fs <> gs then 1.0 else 0.0)
    Statistics.Mean difference

let logistic_learning (labeled_points:(Vector<float>*float) list) =
    let wInit = Vector.Build.Dense(3,0.0)
    let w = learnStatGrad wInit labeled_points
    (w, counter)

let run_experiment N =
//    """
//    Runs an experiment with `N` random points.
//    """
    let linePoints = [for i in 1..2 -> Vector.Build.Random(2, new ContinuousUniform(-1.0,1.0) )]
    let dataPoints = [for i in 1..N -> Vector.Build.Random(2, new ContinuousUniform(-1.0,1.0) )]
    let f = find_line(linePoints.[0], linePoints.[1])
    let labeled_points = dataPoints |> List.map (fun v -> label_point(f, v))
    let (g,num_iterations) = logistic_learning labeled_points
    let p = calculateP(f, g)
    num_iterations



[<EntryPoint>]
let main argv =
    //let iterCount = gradDesc (1.0,1.0) 0.1
    //let error = coordDesc (1.0,1.0) 0.1

    let N = 100
    let results = [for i in [1..1] -> run_experiment N]

    printfn "%A" error //iterCount
    0 // return an integer exit code
