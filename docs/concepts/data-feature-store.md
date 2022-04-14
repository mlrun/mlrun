# Data stores and feature store

One of the biggest challenge in distributed systems is handling data given the different access methods, APIs, and 
authentication mechanisms across types and providers.

MLRun provides three main abstractions to access structured and unstructured data:

- [Data Store](../store/datastore) &mdash; defines a storage provider (e.g. file system, S3, Azure blob, Iguazio v3io, etc.)
- [Data items](../concepts/data-items) &mdash; represent a data item or collection of such (file, dir, table, etc.)
- [Artifacts](../store/artifacts) &mdash; Metadata describing one or more data items. see Artifacts.

Working with the abstractions enable you to securely access different data sources through a single API, many continuance methods (e.g. to/from DataFrame, get, download, list, ..), automated data movement, and versioning.

**In this section**
```{toctree}
:maxdepth: 2
../store/datastore
data-items
../feature-store/feature-store
```