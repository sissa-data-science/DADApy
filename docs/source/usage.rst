Typical usage of the package
============================

A typical usage of Duly involves the initialisation of the Data class with ...

.. code-block:: python
	
	import numpy as np
    from duly.data import Data 

	X = np.random.normal(0, 1, (1000))
	
    data = Data(X)

	...
