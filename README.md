# nativebb optimises a few brainbow operations (C/OpenMP) 

These functions help speed-up some operations required by libbrainbow.

For now the following (optimized) functions are available:

* **dedupcol()** and **dedupcol__indexes()** finds unique triplet colours in a (n x 3) array. Then returns either a new (m x 3) array or a (m x 1) array of unique indexes.
* **isinside()** takes an array of (n x 2) points, an array of (n x 2) points defining a closed polygon and returns a (m x 1) array of the indexes of the points inside the polygon.

