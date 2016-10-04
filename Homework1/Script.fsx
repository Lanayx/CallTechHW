#load @"..\packages\FSharp.Charting.0.90.14\FSharp.Charting.fsx"
#load @"..\packages\MathNet.Numerics.FSharp.3.13.1\MathNet.Numerics.fsx"

open System
open FSharp.Charting
open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double
open MathNet.Numerics.Distributions


let N = 100
let linePoints = [|for i in 1..2 -> Vector.Build.Random(2, new ContinuousUniform(-1.0,1.0) )|]
let dataPoints = [for i in 1..N -> Vector.Build.Random(2, new ContinuousUniform(-1.0,1.0) )]


let getrandomitem() =
  let rnd = System.Random()
  fun (combos : string list) -> List.item (rnd.Next(combos.Length)) combos

let label_point(line: Vector<float>, point: Vector<float>) =
    if line*point > 0.0 then (point,1.0) else (point,-1.0)

let find_line(point1: Vector<float>, point2: Vector<float>) =

//    Finds the line crossing `point_1` and `point_2`. Returns an array
//    [w0, w1, w2] representing a line: w0 + w1 * x + w2 * y = 0.
//    `point_1` and `point_2` are 2-element arrays.

//    # [x1 y1] . [w1] = [-w0] ==> a . x = b
//    # [x2 y2]   [w2]   [-w0]
    let w0 = 1.0
    let (w2, w1) = Fit.Line([|point1.[0]; point2.[0]|], [|point1.[1]; point2.[1]|])
    Vector.Build.Dense([|w0; w1; w2|])




let perceptron_learning correctLabels =
    let wInit = Vector.Build.Dense(3,0.0)
    let classified =  dataPoints |> List.map (fun v -> label_point(wInit, v))
    let mskl = List.zip correctLabels classified
                              |> List.filter (fun(y,(x,s)) -> y <> s)
                              |> List.map (fun (y, x) -> x)
    let mutable counter = 0

    let rec learn (misclasified: (Vector<float>*float) list) (w: Vector<float>) =
        match misclasified with
            | [] -> None
            | (x,s)::t ->
                counter <- counter + 1
                let newW = w + x * Vector.Build.Dense([|1.0;x.[0];x.[1]|])
                let classifiedNew =  dataPoints |> List.map (fun v -> label_point(newW, v))
                let msklNew = List.zip correctLabels classifiedNew
                               |> List.filter (fun(y,(x,s)) -> y <> s)
                               |> List.map (fun (y, x) -> x)
                learn msklNew newW
    learn mskl wInit
    counter

let run_experiment(N) =
//    """
//    Runs an experiment with `N` random points.
//    """
    let f = find_line(linePoints.[0], linePoints.[1])
    let correct_labels = dataPoints |> List.map (fun v -> label_point(f, v))
    let num_iterations = perceptron_learning (List.map (fun (point, s) -> s) correct_labels)
    num_iterations

run_experiment(10)


//let main()=
//    for N in [10;100] do //  # N is the number of sample points
//        results = [run_experiment(N) for _ in range(NUM_RUNS)]
//        num_iterations, p_f_neq_g = np.mean(results, axis=0)
//        print('N={}: iter={}, prob={}'.format(N, num_iterations, p_f_neq_g))



//
//Chart.Combine(
//    [ Chart.Line(linePoints |> List.map (fun (v) -> (v.[0], v.[1])))
//      Chart.Point(data = (dataPoints |> List.map (fun (v,i) -> (v.[0], v.[1]))),
//                    Labels = (pointsWithSigns |> List.map (fun (v,i, sign) -> String.Format("{0}({1})",i, sign))))])
