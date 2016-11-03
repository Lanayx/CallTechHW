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

let swap (a: _[]) x y =
    let tmp = a.[x]
    a.[x] <- a.[y]
    a.[y] <- tmp

// shuffle an array (in-place)
let shuffle arr =
    let a = List.toArray arr
    Array.iteri (fun i _ -> swap a i (rand.Next(i, Array.length a))) a
    Array.toList a

let Ein (values : (Vector<float>*float) list) (w: Vector<float>) =
    Statistics.Mean [ for (xn,yn) in values -> log(1.0 + exp(-1.0*yn*(w*xn)))]

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
    | x when x.L2Norm() < 0.01 -> w
    | _ -> learnFullGrad (w+deltaW) labeled_points

let rec learnStatGrad (w: Vector<float>) (labeled_points: (Vector<float>*float) list) =
    match labeled_points with
    | [] -> w
    | (head::tail) ->
        let (xn, yn) = head
        let grad = yn*xn /( 1.0 + exp(yn*(w*xn)))
        let deltaW = nu*grad
        learnStatGrad (w+deltaW) tail

let rec learnStatGradCover w labeled_points =
    counter <- counter + 1
    let newW = learnStatGrad w labeled_points
    let deltaNorm = (newW - w).L2Norm();
    match deltaNorm with
    | x when x < 0.01 -> newW
    | _ -> learnStatGradCover newW (labeled_points |> shuffle)

let logistic_learning (labeled_points:(Vector<float>*float) list) =
    let wInit = Vector.Build.Dense(3,0.0)
    counter <- 0
    let w = learnStatGradCover wInit labeled_points
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
    let p = Ein labeled_points g
    (num_iterations,p)



[<EntryPoint>]
let main argv =
    //let iterCount = gradDesc (1.0,1.0) 0.1
    //let error = coordDesc (1.0,1.0) 0.1

    let N = 100
    let results = [for i in [1..100] -> run_experiment N]
    let numAv = Statistics.Mean (List.map ( fun (n,p) -> (float) n) results)
    let errAv = Statistics.Mean (List.map ( fun (n,p) -> p) results)

    printfn "%A" results //iterCount
    0 // return an integer exit code
