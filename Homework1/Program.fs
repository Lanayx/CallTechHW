open System
open FSharp.Charting
open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double
open MathNet.Numerics.Distributions

let N = 10
let mutable counter = 0
let linePoints = [|for i in 1..2 -> Vector.Build.Random(2, new ContinuousUniform(-1.0,1.0) )|]
let dataPoints = [for i in 1..N -> Vector.Build.Random(2, new ContinuousUniform(-1.0,1.0) )]


let getrandomitem() =
  let rnd = System.Random()
  fun (combos : string list) -> List.nth combos (rnd.Next(combos.Length))

let label_point(line: Vector<float>, point: Vector<float>) =
    let point3 = Vector.Build.Dense([|1.0;point.[0];point.[1]|])
    if line*point3 > 0.0 then (point,1.0) else (point,-1.0)

let find_line(point1: Vector<float>, point2: Vector<float>) =

//    """
//    Finds the line crossing `point_1` and `point_2`. Returns an array
//    [w0, w1, w2] representing a line: w0 + w1 * x + w2 * y = 0.
//    `point_1` and `point_2` are 2-element arrays.
//    """
//    # [x1 y1] . [w1] = [-w0] ==> a . x = b
//    # [x2 y2]   [w2]   [-w0]
//look at http://numerics.mathdotnet.com/LinearEquations.html

    a = [point_1, point_2]
    w0 = 1.
    b = [-w0, -w0]
    w1, w2 = np.linalg.solve(a, b)
    return np.array([w0, w1, w2])


let rec learn (misclasified: (Vector<float>*float) list) (w: Vector<float>) correctLabels =
        match misclasified with
            | [] -> None
            | (x,s)::t ->
                counter <- counter + 1
                let newW = w + Vector.Build.Dense([|1.0;x.[0];x.[1]|])*s
                let classifiedNew =  dataPoints |> List.map (fun v -> label_point(newW, v))
                let msklNew = List.zip correctLabels classifiedNew
                               |> List.filter (fun(y,(x,s)) -> y <> s)
                               |> List.map (fun (y, x) -> x)
                learn msklNew newW correctLabels

let perceptron_learning correctLabels =
    let wInit = Vector.Build.Dense(3,0.0)
    let classified =  dataPoints |> List.map (fun v -> label_point(wInit, v))
    let mskl = List.zip correctLabels classified
                              |> List.filter (fun(y,(x,s)) -> y <> s)
                              |> List.map (fun (y, (x, s)) -> (x,y))

    learn mskl wInit correctLabels
    counter

let run_experiment(N) =
//    """
//    Runs an experiment with `N` random points.
//    """
    let f = find_line(linePoints.[0], linePoints.[1])
    let correct_labels = dataPoints |> List.map (fun v -> label_point(f, v))
    let num_iterations = perceptron_learning (List.map (fun (point, s) -> s) correct_labels)
    num_iterations

[<EntryPoint>]
let main argv =
    run_experiment(10)

    0
