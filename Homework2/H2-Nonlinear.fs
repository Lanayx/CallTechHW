module NonLinear

open System
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double
open MathNet.Numerics.Distributions
open MathNet.Numerics.Statistics


let rand = new Random()

let init N =
    let dataPoints = [for i in 1..N -> Vector.Build.Random(2, new ContinuousUniform(-1.0,1.0) )]
    let linePoints = [ for i in 1..2 -> Vector.Build.Random(2, new ContinuousUniform(-1.0,1.0) )]
    (linePoints, dataPoints)

let label_point(line: Vector<float>, point: Vector<float>) =
    //let point3 = Vector.Build.Dense([|1.0;point.[0];point.[1]|])
    let (x1,x2) = (point.[0],point.[1])
    let point3 = vector [1.0; x1; x2; x1*x2; x1*x1; x2*x2 ]

    let mult = point3*line
    if  mult > 0.0
    then (point,1.0)
    else (if mult = 0.0 then (point,0.0) else (point,-1.0))



let target_function(point: Vector<float>) =
    let result = point.[0]*point.[0] + point.[1]*point.[1] - 0.6
    match result with
    | a when a > 0.0 -> (point, 1.0)
    | a when a = 0.0 -> (point, 0.0)
    | _ -> (point, -1.0)

let regression correct_labels (dataPoints: Vector<float> list) =
//    let normalizedPoints = dataPoints |> List.map (fun v -> [|1.0; v.[0]; v.[1]|])
    let normalizedPoints = dataPoints |>
                           List.map (fun v -> [|1.0; v.[0]; v.[1]; v.[0]* v.[1]; v.[0]*v.[0]; v.[1]*v.[1] |])

    let x = Matrix.Build.DenseOfRowArrays(normalizedPoints)
    let y = Vector.Build.DenseOfEnumerable(correct_labels).ToColumnMatrix()
    let xT = x.Transpose()
    let w = (xT*x).Inverse()*xT*y
    vector [w.[0,0]; w.[1,0]; w.[2,0]; w.[3,0]; w.[4,0]; w.[5,0]]

let calculateP correct_labels dataPoints g =
    let points = dataPoints
    let label_g = points |> List.map (fun v -> label_point(g, v))
    let difference = List.zip correct_labels label_g |> List.map (fun(fs,(g,gs)) -> if fs <> gs then 1.0 else 0.0)
    Statistics.Mean difference

let generate_noize s =
    let a = rand.Next(10);
    if a>0 then s else -s


let run() =
    let (linePoints, dataPoints) = init 1000

    let correct_labels = dataPoints
                         |> List.map (fun v -> target_function(v))
                         |> List.map (fun (x,s) -> s)
                         |> List.map generate_noize
    let g = regression correct_labels dataPoints
    let p = calculateP correct_labels dataPoints g

    let (_, newDataPoints) = init 1000

    let new_correct_labels = newDataPoints
                             |> List.map (fun v -> target_function(v))
                             |> List.map (fun (x,s) -> s)
                             |> List.map generate_noize
    let p1 = calculateP new_correct_labels newDataPoints g
    (g, p, p1)



