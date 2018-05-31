let name = "isaac"

let age = 38
let person = name, age

open System.IO
let projects =
    Directory.EnumerateDirectories @"C:\users\isaac\source\repos"
    |> Seq.map (DirectoryInfo >> fun d -> d.Name)
    |> Seq.toArray