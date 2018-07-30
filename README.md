
# Example of using ML.NET from F# Interactive Scripting

Windows:

    .paket\paket.exe update

Unix:

    mono .paket/paket.exe update

Then edit and execute [mlnet.fsx](mlnet.fsx).

In [mlfs.fsx](mlfs.fsx) there are samples of how to wrap the ML.NET API in F# helpers. There is no plan to add these to ML.NET directly, they are just there for demonstration.

The use of classes for data is needed because of [this issue with ML.NET](https://github.com/dotnet/machinelearning/issues/180).

# Notes

* Author: @isaacabraham for ML.NET 0.1

* Updated for ML.NET 0.3 by @dsyme

