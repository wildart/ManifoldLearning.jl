# Datasets

```@setup EG
using Plots, ManifoldLearning
gr(fmt=:svg)
```

The __ManifoldLearning__ package provides an implementation of synthetic test datasets:

```@docs
ManifoldLearning.swiss_roll
```
```@example EG
X, L = ManifoldLearning.swiss_roll(segments=5); #hide
scatter3d(X[1,:], X[2,:], X[3,:], c=L.+2, palette=cgrad(:default), ms=2.5, leg=:none, camera=(10,10)) #hide
```

```@docs
ManifoldLearning.spirals
```
```@example EG
X, L = ManifoldLearning.spirals(segments=5); #hide
scatter3d(X[1,:], X[2,:], X[3,:], c=L.+2, palette=cgrad(:default), ms=2.5, leg=:none, camera=(10,10)) #hide
```

```@docs
ManifoldLearning.scurve
```
```@example EG
X, L = ManifoldLearning.scurve(segments=5); #hide
scatter3d(X[1,:], X[2,:], X[3,:], c=L.+2, palette=cgrad(:default), ms=2.5, leg=:none, camera=(10,10)) #hide
```
