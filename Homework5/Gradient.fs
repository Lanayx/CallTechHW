// Learn more about F# at http://fsharp.org
// See the 'F# Tutorial' project for more help.
module Gradient
let mutable counter = 0
let error (u: double) (v: double) =
    (u*exp(v)-2.0*v*exp(-u))**2.0

let dEdu (u: double) (v: double) =
    2.0*(exp(v)+2.0*v*exp(-u))*(u*exp(v)-2.0*v*exp(-u))

let dEdv (u: double) (v: double) =
    2.0*(u*exp(v)-2.0*exp(-u))*(u*exp(v)-2.0*v*exp(-u))

let rec gradDesc (u,v) nu =
    counter <- counter + 1
    let (du, dv) = (dEdu u v, dEdv u v)
    let (newU, newV) = (u-nu*du, v-nu*dv)
    let errorRes = error newU newV
    match errorRes with
    | e when e < 0.00000000000001 -> counter
    | _ -> gradDesc (newU, newV) nu

let rec coordDesc (u,v) nu =
    counter <- counter + 1
    let du = dEdu u v
    let newU = u-nu*du
    let dv = dEdv newU v
    let newV = v-nu*dv
    match counter with
    | 15 -> error newU newV
    | _ -> coordDesc (newU, newV) nu
