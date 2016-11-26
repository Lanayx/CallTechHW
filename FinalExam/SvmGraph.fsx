#load @"..\packages\FSharp.Charting.0.90.14\FSharp.Charting.fsx"
#load @"..\packages\MathNet.Numerics.FSharp.3.14.0-beta03\MathNet.Numerics.fsx"

let data = [((1.0,0.0),-1.0), ((0.0,1.0),-1.0), ((0.0,-1.0),-1.0),
            ((-1.0,0.0),1.0), ((0.0,2.0),1.0), ((0.0,-2.0),1.0),
            ((-2.0,0.0),1.0)]

let transform (x1,x2) =
    (x2*x2-2.0*x1-1.0,x1*x1-2.0*x2+1.0)