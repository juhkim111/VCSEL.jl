# API 

```@meta
CurrentModule = VCSEL
```

## Constructing models 

```@docs 
VCModel
VCintModel 
```

## Properties of model 

```@docs 
length(::VCModel)
ncovariates
size(::VCModel)
nmeanparams(::VCModel)
nvarcomps(::VCModel)
ngroups(::VCModel)
```


## Fitting model 

```@docs 
vcselect!
vcselectpath!
```

## Utilities 

```@docs 
resetModel!
maxlambda 
rankvarcomps
plotsolpath 
```
