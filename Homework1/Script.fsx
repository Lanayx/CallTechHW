#load @"..\packages\FSharp.Charting.0.90.14\FSharp.Charting.fsx"
#load @"..\packages\MathNet.Numerics.FSharp.3.13.1\MathNet.Numerics.fsx"

open System
open FSharp.Charting
open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double
open MathNet.Numerics.Distributions
open MathNet.Numerics.Statistics


let mutable N = 100
let mutable counter = 0.0
let mutable dataPoints = [for i in 1..N -> Vector.Build.Random(2, new ContinuousUniform(-1.0,1.0) )]
let mutable linePoints =  [|for i in 1..2 -> Vector.Build.Random(2, new ContinuousUniform(-1.0,1.0) )|]
let rand = new Random()

let getrandomitem() =
  let rnd = System.Random()
  fun (combos : string list) -> List.nth combos (rnd.Next(combos.Length))

let label_point(line: Vector<float>, point: Vector<float>) =
    let point3 = Vector.Build.Dense([|1.0;point.[0];point.[1]|])
    let mult = point3*line
    if  mult > 0.0
    then (point,1.0)
    else (if mult = 0.0 then (point,0.0) else (point,-1.0))

let find_line(point1: Vector<float>, point2: Vector<float>) =

//    """
//    Finds the line crossing `point_1` and `point_2`. Returns an array
//    [w0, w1, w2] representing a line: w0 + w1 * x + w2 * y = 0.
//    `point_1` and `point_2` are 2-element arrays.
//    """
//    # [x1 y1] . [w1] = [-w0] ==> a . x = b
//    # [x2 y2]   [w2]   [-w0]

    let a = Matrix<float>.Build.DenseOfRowArrays( [| point1.ToArray(); point2.ToArray() |])
    let w0 = 1.0
    let b = vector [-w0; -w0]
    let result = a.Solve(b).ToArray();
    vector [w0; result.[0]; result.[1]]


let rec learn (misclasified: (Vector<float>*float) list) (w: Vector<float>) correctLabels =
        match misclasified with
            | [] -> w
            | (x,s)::t ->
                counter <- counter + 1.0
                let newW = w + Vector.Build.Dense([|1.0;x.[0];x.[1]|])*s
                let classifiedNew =  dataPoints |> List.map (fun v -> label_point(newW, v))
                let msklNew = List.zip correctLabels classifiedNew
                               |> List.filter (fun(y,(x,s)) -> y <> s)
                               |> List.map (fun (y, (x, s)) -> (x,y))
                               |> List.sortWith(fun elem elem2 -> if rand.Next() > rand.Next() then 1 else -1 )
                learn msklNew newW correctLabels

let perceptron_learning correctLabels =
    let wInit = Vector.Build.Dense(3,0.0)
    let classified =  dataPoints |> List.map (fun v -> label_point(wInit, v))
    let mskl = List.zip correctLabels classified
                              |> List.filter (fun(y,(x,s)) -> y <> s)
                              |> List.map (fun (y, (x, s)) -> (x,y))
                              |> List.sortWith(fun elem elem2 -> if rand.Next() > rand.Next() then 1 else -1 )

    let w = learn mskl wInit correctLabels
    (w, counter)

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

let run_experiment() =
//    """
//    Runs an experiment with `N` random points.
//    """
    counter <- 1.0
    linePoints <- [|for i in 1..2 -> Vector.Build.Random(2, new ContinuousUniform(-1.0,1.0) )|]
    dataPoints <- [for i in 1..N -> Vector.Build.Random(2, new ContinuousUniform(-1.0,1.0) )]
    let f = find_line(linePoints.[0], linePoints.[1])
    let correct_labels = dataPoints |> List.map (fun v -> label_point(f, v))
    let (g,num_iterations) = perceptron_learning (List.map (fun (point, s) -> s) correct_labels)
    let p = calculateP(f, g)
    (num_iterations, p)

run_experiment()


//let main()=
//    for N in [10;100] do //  # N is the number of sample points
//        results = [run_experiment(N) for _ in range(NUM_RUNS)]
//        num_iterations, p_f_neq_g = np.mean(results, axis=0)
//        print('N={}: iter={}, prob={}'.format(N, num_iterations, p_f_neq_g))




Chart.Combine(
    [ Chart.Line(linePoints |> Array.map (fun v -> (v.[0], v.[1])))
      Chart.Point(data = (dataPoints |> List.map (fun v -> (v.[0], v.[1]))))])
