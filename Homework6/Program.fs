open System.IO
open System
open MathNet.Numerics.LinearAlgebra

// Learn more about F# at http://fsharp.org
// See the 'F# Tutorial' project for more help.

let regression (dataPoints: (float*float) list) correct_labels  =
    let normalizedPoints = dataPoints |>
                           List.map (fun (x1,x2) -> [|1.0; x1; x2; x1*x1; x2*x2; x1*x2; abs(x1-x2); abs(x1+x2) |])

    let x = Matrix.Build.DenseOfRowArrays(normalizedPoints)
    let y = Vector.Build.DenseOfEnumerable(correct_labels)
    let xT = x.Transpose()
    let w = (xT*x).Inverse()*xT*y
    w

let readData (fileName: string) =
    let values = seq {
            use sr = new StreamReader(fileName)
            while not sr.EndOfStream do
                yield sr.ReadLine()
        }
    values
         |> Seq.map (fun line -> line.Split([|' '|], StringSplitOptions.RemoveEmptyEntries))
         |> Seq.map (fun arr -> (((float)arr.[0],(float)arr.[1]), (float)arr.[2]))
         |> Seq.toList

let einCalc (dataPoints: (float*float) list) (correct_labels: float list) (w: Vector<float>) =
    let normalizedPoints = dataPoints |>
                           List.map (fun (x1,x2) -> [|1.0; x1; x2; x1*x1; x2*x2; x1*x2; abs(x1-x2); abs(x1+x2) |])
    let x = Matrix.Build.DenseOfRowArrays(normalizedPoints)
    let y = Vector.Build.DenseOfEnumerable(correct_labels)
    1.0/((float)dataPoints.Length)*(((x*w - y).L2Norm())**2.0)


[<EntryPoint>]
let main argv =
    let testData = readData "../../in.dta"
    let testX = testData |> List.map (fun (x,y) -> x)
    let testY = testData |> List.map (fun (x,y) -> y)
    let w = regression testX testY
    let ein = einCalc testX testY w
    printfn "%A" argv
    0 // return an integer exit code
