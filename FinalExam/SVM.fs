module SVM

open LibSVMsharp

let data = [((1.0,0.0),-1.0); ((0.0,1.0),-1.0); ((0.0,-1.0),-1.0);
            ((-1.0,0.0),1.0); ((0.0,2.0),1.0); ((0.0,-2.0),1.0);
            ((-2.0,0.0),1.0)]

let SVMRun() =
    let problem = new SVMProblem()
    problem.X.AddRange(data |> List.map (fun ((x1,x2),y) -> [| SVMNode(1,x1); SVMNode(2,x2) |]))
    problem.Y.AddRange(data |> List.map (fun ((x1,x2),y) -> y))

    let parameter = new SVMParameter();
    parameter.Type <- SVMType.C_SVC;
    parameter.Kernel <- SVMKernelType.POLY;
    parameter.Degree <- 2
    parameter.C <- 10000.0;
    parameter.Coef0 <- 1.0;
    parameter.Gamma <- 1.0;

    let model = SVM.Train(problem, parameter);
    let results = data |> List.map (fun ((x1,x2),y) -> SVM.Predict(model, [| SVMNode(1,x1); SVMNode(2,x2) |]))
    let labels = data |> List.map (fun ((x1,x2),y) -> y)
    let ein = List.sum (List.zip results labels |> List.map (fun (res,lab) -> if res = lab then 0.0 else 1.0)) / (float)labels.Length

    printfn "%A" model.SV