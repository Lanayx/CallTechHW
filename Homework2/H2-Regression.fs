module Regression

open System
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double
open MathNet.Numerics.Distributions
open MathNet.Numerics.Statistics


let rand = new Random()

let init N =
    let dataPoints = [for i in 1..N -> Vector.Build.Random(2, new ContinuousUniform(-1.0,1.0) )]
    let linePoints = [ for i in 1..2 -> Vector.Build.Random(2, new ContinuousUniform(-1.0,1.0) )]
//    let linePoints = [ Vector.Build.Dense([|-0.40877; -0.653459|]);
//                           Vector.Build.Dense([|-0.579892;  -0.84767|])];
//
//
//    let dataPoints = [
//                           Vector.Build.Dense([|0.308796; -0.758407|]);
//                           Vector.Build.Dense([|-0.173793; 0.120863|]);
//                           Vector.Build.Dense([|0.841236; -0.780489|]);
//                           Vector.Build.Dense([|-0.291265; 0.129208|]);
//                           Vector.Build.Dense([|-0.812164; -0.697282|]); ]



    (linePoints, dataPoints)

let label_point(line: Vector<float>, point: Vector<float>) =
    let point3 = Vector.Build.Dense([|1.0;point.[0];point.[1]|])
    let mult = point3*line
    if  mult > 0.0
    then (point,1.0)
    else (if mult = 0.0 then (point,0.0) else (point,-1.0))

let find_line(point1: Vector<float>, point2: Vector<float>) =
    let a = Matrix<float>.Build.DenseOfRowArrays( [| point1.ToArray(); point2.ToArray() |])
    let w0 = 1.0
    let b = vector [-w0; -w0]
    let result = a.Solve(b).ToArray()
    vector [w0; result.[0]; result.[1]]

let regression correct_labels (dataPoints: Vector<float> list) =
    let normalizedPoints = dataPoints |> List.map (fun v -> [|1.0; v.[0]; v.[1]|])
    let x = Matrix.Build.DenseOfRowArrays(normalizedPoints)
    let y = Vector.Build.DenseOfEnumerable(correct_labels).ToColumnMatrix()
    let xT = x.Transpose()
    let w = ((xT*x).Inverse())*xT*y
    vector [w.[0,0]; w.[1,0]; w.[2,0]]

let calculateP (f, g) dataPoints =
    let points = dataPoints
    let label_f = points |> List.map (fun v -> label_point(f, v))
    let label_g = points |> List.map (fun v -> label_point(g, v))
    let difference = List.zip label_f label_g |> List.map (fun((f,fs),(g,gs)) -> if fs <> gs then 1.0 else 0.0)
    Statistics.Mean difference


let mutable counter = 0.0
let rec learn (misclasified: (Vector<float>*float) list) (w: Vector<float>) correctLabels dataPoints =
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
                learn msklNew newW correctLabels dataPoints

let perceptron_learning correctLabels dataPoints wInit =
    let classified =  dataPoints |> List.map (fun v -> label_point(wInit, v))
    let mskl = List.zip correctLabels classified
                              |> List.filter (fun(y,(x,s)) -> y <> s)
                              |> List.map (fun (y, (x, s)) -> (x,y))
                              |> List.sortWith(fun elem elem2 -> if rand.Next() > rand.Next() then 1 else -1 )

    let w = learn mskl wInit correctLabels dataPoints
    (w, counter)

let runPerceptron correct_labels dataPoints wInit =
    counter <- 1.0
    let (g,num_iterations) = perceptron_learning correct_labels dataPoints wInit
    num_iterations

let run() =
    let (linePoints, dataPoints) = init 10
    let f = find_line(linePoints.[0], linePoints.[1])
    let correct_labels = dataPoints
                         |> List.map (fun v -> label_point(f, v))
                         |> List.map (fun (x,s) -> s)
    let g = regression correct_labels dataPoints
    runPerceptron correct_labels dataPoints g

//    let p = calculateP (f, g) dataPoints
//    let (_, newDataPoints) = init 1000
//    let correct_labels = newDataPoints
//                         |> List.map (fun v -> label_point(f, v))
//                         |> List.map (fun (x,s) -> s)
//    let p1 = calculateP (f, g) newDataPoints
//    (p,p1)



