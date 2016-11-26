#load @"..\packages\FSharp.Charting.0.90.14\FSharp.Charting.fsx"

open System
open FSharp.Charting

let data = [((1.0,0.0),-1.0); ((0.0,1.0),-1.0); ((0.0,-1.0),-1.0);
            ((-1.0,0.0),1.0); ((0.0,2.0),1.0); ((0.0,-2.0),1.0);
            ((-2.0,0.0),1.0)]

let transform (x1,x2) =
    (x2*x2-2.0*x1-1.0,x1*x1-2.0*x2+1.0)

let transformedData = data |>
                      List.map (fun (x,y) -> ((transform x),y))

let getLinePoint x1 (w1,w2) b =
    if w2 = 0.0 then (-b/w1,x1)
    else (x1,-(b+w1*x1)/w2)

Chart.Combine(
    [ Chart.Line( [-5.0..0.5..5.0] |> List.map (fun x1 -> getLinePoint x1 (1.0,0.0) -0.5))
      Chart.Point(data = (transformedData |> List.map (fun (x,y)-> x)),
                 Labels = (transformedData |> List.map (fun (x,y)-> y.ToString())))])

