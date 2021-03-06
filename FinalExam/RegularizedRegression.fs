﻿module RegularizedRegression

open MathNet.Numerics.Statistics
open MathNet.Numerics.LinearAlgebra
open System.IO
open System

let noTransform (x1,x2) =
    [1.0; x1; x2;]

let transform (x1,x2) =
    [1.0; x1; x2; x1*x2; x1*x1; x2*x2]

let weightedRegression (data: ((float*float)*float) list) (lambda:float) transform  =
    let dataPoints = data |> List.map (fun (x,y) -> x)
    let correct_labels = data |> List.map (fun (x,y) -> y)
    let normalizedPoints = dataPoints |> List.map transform
    let x = matrix normalizedPoints
    let y = vector correct_labels
    let xT = x.Transpose()
    let I = Matrix.Build.DenseIdentity(normalizedPoints.Head.Length) :> Matrix<float>
    let w = (xT*x+lambda*I).Inverse()*xT*y
    w

let readData (fileName: string) =
    let values = seq {
            use sr = new StreamReader(fileName)
            while not sr.EndOfStream do
                yield sr.ReadLine()
        }
    values
         |> Seq.map (fun line -> line.Split([|' '|], StringSplitOptions.RemoveEmptyEntries))
         |> Seq.map (fun arr -> (((float)arr.[1],(float)arr.[2]), (float)arr.[0]))
         |> Seq.toList

let one_vs_one data (number1: float) (number2: float) =
    data |> List.filter (fun (x,y) -> y = number1 || y = number2)
         |> List.map (fun (x,y) -> if y = number1 then (x, 1.0) else (x,-1.0))

let one_vs_all data (number: float) =
    data |> List.map (fun (x,y) -> if y = number then (x, 1.0) else (x,-1.0))

let calculateE  (data: ((float*float)*float) list) (w: Vector<float>) transform=
    let correct_labels = data |> List.map (fun (x,y) -> y)
    let errors =
        data
        |> List.map (fun (x, y) -> (vector (transform x), y))
        |> List.map (fun (x, y) -> if (sign (x*w) = (int)y) then 0.0 else 1.0) //fix dot operation
    Statistics.Mean errors

let RRRun() =
    let testData = readData "../../features.train"
    let verificationData = readData "../../features.test"

    let currentTransform = transform
    let testSet = one_vs_one testData 1.0 5.0
    let verificationSet = one_vs_one verificationData 1.0 5.0
    let wS = [0.01;1.0] |> List.map (fun labda -> weightedRegression testSet labda currentTransform)
    let eins = wS |> List.map (fun w -> calculateE testSet w currentTransform)
    let eouts = wS |> List.map (fun w -> calculateE verificationSet w currentTransform)
    printfn "eins: %A eouts:%A " eins eouts
//    let result =[noTransform; transform]
//                |> List.map (fun currentTransform ->
//                    let testSets = [ for i in 0.0..9.0 -> one_vs_all testData i]
//                    let verificationSets = [ for i in 0.0..9.0 -> one_vs_all verificationData i]
//                    let wS = testSets |> List.map (fun testSet -> weightedRegression testSet 1.0 currentTransform)
//                    let eins = List.zip testSets wS |> List.map (fun (set, w) -> calculateE set w currentTransform)
//                    let eouts = List.zip verificationSets wS |> List.map (fun (set, w) -> calculateE set w currentTransform)
//                    (eins, eouts))
//                |> List.map (fun (eins, eouts) -> List.zip eins eouts |> List.map (fun (ein, eout) -> Math.Round(eout - ein,4)))
//    printfn "%A" result