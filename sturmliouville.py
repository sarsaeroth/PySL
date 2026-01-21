import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import legendre, chebyt


class SturmLiouville:
    
    def __init__(self):
        self._p_data = None
        self._q_data = None
        self._w_data = None

        self._x_grid = None
        self._p_grid = None
        self._q_grid = None
        self._w_grid = None

        self._domain = None
        self._n_points = None

        # Boundary condition settings
        self._bc_type = 'dirichlet'
        self._bc_params = {
            'left': {'alpha': 1.0, 'beta': 0.0, 'gamma': 0.0},
            'right': {'alpha': 1.0, 'beta': 0.0, 'gamma': 0.0}
        }

        # Basis function settings
        self._basis_type = 'fourier'
        self._basis_kwargs = {}
        self._custom_basis = None

        # Computed basis functions on grid
        self._basis_functions = None
        self._basis_derivatives = None

    def p(self, function_data):
        self._validate_function_data(function_data, 'p(x)')
        self._p_data = dict(function_data)
        return self

    def q(self, function_data):
        self._validate_function_data(function_data, 'q(x)')
        self._q_data = dict(function_data)
        return self

    def w(self, function_data):
        self._validate_function_data(function_data, 'w(x)')
        self._w_data = dict(function_data)
        return self

    def _validate_function_data(self, data, name):
        if not isinstance(data, dict):
            raise TypeError(f"{name} data must be a dictionary, got {type(data).__name__}")
        if len(data) < 2:
            raise ValueError(f"{name} data must have at least 2 points")
        for x, y in data.items():
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                raise TypeError(f"{name} data keys and values must be numeric")
            if np.isnan(x) or np.isnan(y) or np.isinf(x) or np.isinf(y):
                raise ValueError(f"{name} data contains NaN or Inf values")

    def set_boundary_conditions(self, boundary_type, **kwargs):
        valid_types = ['dirichlet', 'neumann', 'mixed']
        if boundary_type not in valid_types:
            raise ValueError(f"boundary_type must be one of {valid_types}, got '{boundary_type}'")

        self._bc_type = boundary_type

        if boundary_type == 'dirichlet':
            # y(a) = 0, y(b) = 0
            self._bc_params = {
                'left': {'alpha': 1.0, 'beta': 0.0, 'gamma': 0.0},
                'right': {'alpha': 1.0, 'beta': 0.0, 'gamma': 0.0}
            }
        elif boundary_type == 'neumann':
            # y'(a) = 0, y'(b) = 0
            self._bc_params = {
                'left': {'alpha': 0.0, 'beta': 1.0, 'gamma': 0.0},
                'right': {'alpha': 0.0, 'beta': 1.0, 'gamma': 0.0}
            }
        elif boundary_type == 'mixed':
            self._bc_params = {
                'left': {
                    'alpha': kwargs.get('alpha_left', 1.0),
                    'beta': kwargs.get('beta_left', 0.0),
                    'gamma': kwargs.get('gamma_left', 0.0)
                },
                'right': {
                    'alpha': kwargs.get('alpha_right', 1.0),
                    'beta': kwargs.get('beta_right', 0.0),
                    'gamma': kwargs.get('gamma_right', 0.0)
                }
            }

        return self

    def set_basis(self, basis_type, **kwargs):
        valid_types = ['polynomial', 'fourier', 'custom']
        if basis_type not in valid_types:
            raise ValueError(f"basis_type must be one of {valid_types}, got '{basis_type}'")

        self._basis_type = basis_type
        self._basis_kwargs = kwargs
        return self

    def set_custom_basis(self, functions):
        if not isinstance(functions, (list, tuple)):
            raise TypeError("functions must be a list or tuple of callables")
        if len(functions) < 1:
            raise ValueError("Must provide at least one basis function")
        for i, f in enumerate(functions):
            if not callable(f):
                raise TypeError(f"Function {i} is not callable")

        self._basis_type = 'custom'
        self._custom_basis = list(functions)
        return self

    def _determine_common_domain(self):
        p_x = np.array(sorted(self._p_data.keys()))
        q_x = np.array(sorted(self._q_data.keys()))
        w_x = np.array(sorted(self._w_data.keys()))

        # Common domain is intersection of all ranges
        a = max(p_x.min(), q_x.min(), w_x.min())
        b = min(p_x.max(), q_x.max(), w_x.max())

        if a >= b:
            raise ValueError("No common domain found for p(x), q(x), w(x) data")

        return (a, b)

    def _compute_minimal_spacing(self):
        spacings = []
        for data in [self._p_data, self._q_data, self._w_data]:
            x_sorted = np.array(sorted(data.keys()))
            if len(x_sorted) > 1:
                diffs = np.diff(x_sorted)
                spacings.append(diffs.min())

        return min(spacings) if spacings else 0.01

    def _interpolate_to_grid(self, data, x_grid):
        x_data = np.array(sorted(data.keys()))
        y_data = np.array([data[x] for x in sorted(data.keys())])

        # Create cubic spline interpolator
        cs = CubicSpline(x_data, y_data, extrapolate=False)

        # Evaluate on grid
        return cs(x_grid)

    def preprocess(self, n_points=None):
        # Check that all data is provided
        if self._p_data is None:
            raise ValueError("p(x) data not set. Call p() first.")
        if self._q_data is None:
            raise ValueError("q(x) data not set. Call q() first.")
        if self._w_data is None:
            raise ValueError("w(x) data not set. Call w() first.")

        # Determine common domain
        self._domain = self._determine_common_domain()
        a, b = self._domain

        # Determine number of points
        if n_points is None:
            min_spacing = self._compute_minimal_spacing()
            n_points = max(50, int(np.ceil((b - a) / min_spacing)) + 1)
        self._n_points = n_points

        # Create uniform grid
        self._x_grid = np.linspace(a, b, n_points)

        # Interpolate functions to grid
        self._p_grid = self._interpolate_to_grid(self._p_data, self._x_grid)
        self._q_grid = self._interpolate_to_grid(self._q_data, self._x_grid)
        self._w_grid = self._interpolate_to_grid(self._w_data, self._x_grid)

        # Validate positivity constraints
        if np.any(self._p_grid <= 0):
            raise ValueError("p(x) must be strictly positive on the domain")
        if np.any(self._w_grid <= 0):
            raise ValueError("w(x) must be strictly positive on the domain")

        return self

    def _generate_polynomial_basis(self, n_basis):
        a, b = self._domain
        poly_type = self._basis_kwargs.get('polynomial_type', 'legendre')

        # Map x to [-1, 1] for standard polynomials
        x_normalized = 2 * (self._x_grid - a) / (b - a) - 1

        basis = []
        for n in range(1, n_basis + 1):
            if poly_type == 'legendre':
                P_n = legendre(n)
                phi = P_n(x_normalized)
            elif poly_type == 'chebyshev':
                T_n = chebyt(n)
                phi = T_n(x_normalized)
            else:
                raise ValueError(f"Unknown polynomial type: {poly_type}")

            basis.append(phi)

        return np.array(basis)

    def _generate_fourier_basis(self, n_basis):
        a, b = self._domain
        L = b - a

        basis = []
        for n in range(1, n_basis + 1):
            if self._bc_type == 'dirichlet':
                # Sine functions vanish at boundaries
                phi = np.sin(n * np.pi * (self._x_grid - a) / L)
            elif self._bc_type == 'neumann':
                # Cosine functions have zero derivative at boundaries
                phi = np.cos(n * np.pi * (self._x_grid - a) / L)
            else:
                # Mixed: use sine as default, apply BC correction later
                phi = np.sin(n * np.pi * (self._x_grid - a) / L)

            basis.append(phi)

        return np.array(basis)

    def _generate_custom_basis(self, n_basis):
        if self._custom_basis is None:
            raise ValueError("Custom basis functions not set. Call set_custom_basis() first.")

        n_available = len(self._custom_basis)
        if n_basis > n_available:
            raise ValueError(f"Requested {n_basis} basis functions but only {n_available} provided")

        basis = []
        for i in range(n_basis):
            phi = self._custom_basis[i](self._x_grid)
            basis.append(phi)

        return np.array(basis)

    def _apply_bc_to_basis(self, basis):
        if self._bc_type == 'dirichlet':
            if self._basis_type == 'polynomial':
                a, b = self._domain
                weight = (self._x_grid - a) * (b - self._x_grid)
                weight = weight / weight.max()  # Normalize
                basis = basis * weight[np.newaxis, :]

        elif self._bc_type == 'neumann':
            pass

        elif self._bc_type == 'mixed':
            pass

        return basis

    def _compute_basis_derivatives(self, basis):
        dx = self._x_grid[1] - self._x_grid[0]
        derivatives = np.zeros_like(basis)

        derivatives[:, 1:-1] = (basis[:, 2:] - basis[:, :-2]) / (2 * dx)

        derivatives[:, 0] = (basis[:, 1] - basis[:, 0]) / dx
        derivatives[:, -1] = (basis[:, -1] - basis[:, -2]) / dx

        return derivatives

    def _integrate(self, y):
        dx = self._x_grid[1] - self._x_grid[0]
        return np.trapz(y, dx=dx)

    def generate_basis(self, n_basis):
        if self._x_grid is None:
            raise ValueError("Must call preprocess() before generating basis")

        if self._basis_type == 'polynomial':
            basis = self._generate_polynomial_basis(n_basis)
        elif self._basis_type == 'fourier':
            basis = self._generate_fourier_basis(n_basis)
        elif self._basis_type == 'custom':
            basis = self._generate_custom_basis(n_basis)
        else:
            raise ValueError(f"Unknown basis type: {self._basis_type}")

        # Apply boundary conditions
        basis = self._apply_bc_to_basis(basis)
        self._basis_functions = basis
        self._basis_derivatives = self._compute_basis_derivatives(basis)

        return basis.copy()

    def rayleigh_quotient(self, y):
        if self._x_grid is None:
            raise ValueError("Must call preprocess() before computing Rayleigh quotient")

        y = np.asarray(y)
        if y.shape != self._x_grid.shape:
            raise ValueError(f"y must have shape {self._x_grid.shape}, got {y.shape}")

        # Compute derivative using central differences
        dx = self._x_grid[1] - self._x_grid[0]
        dy = np.zeros_like(y)
        dy[1:-1] = (y[2:] - y[:-2]) / (2 * dx)
        dy[0] = (y[1] - y[0]) / dx
        dy[-1] = (y[-1] - y[-2]) / dx

        # Compute integrands
        numerator_integrand = self._p_grid * dy**2 + self._q_grid * y**2
        denominator_integrand = self._w_grid * y**2

        # Integrate
        numerator = self._integrate(numerator_integrand)
        denominator = self._integrate(denominator_integrand)

        if abs(denominator) < 1e-15:
            raise ValueError("Denominator of Rayleigh quotient is zero (trivial solution)")

        return numerator / denominator

    def rayleigh_quotient_from_coefficients(self, coefficients):
        if self._basis_functions is None:
            raise ValueError("Must call generate_basis() before using coefficients")

        coefficients = np.asarray(coefficients)
        if len(coefficients) > len(self._basis_functions):
            raise ValueError(f"Too many coefficients: got {len(coefficients)}, "
                           f"but only {len(self._basis_functions)} basis functions")

        # Reconstruct trial function from coefficients
        y = np.zeros_like(self._x_grid)
        for i, c in enumerate(coefficients):
            y += c * self._basis_functions[i]

        return self.rayleigh_quotient(y)

    def get_grid(self):
        if self._x_grid is None:
            raise ValueError("Must call preprocess() before getting grid")
        return self._x_grid.copy()

    def get_domain(self):
        if self._domain is None:
            raise ValueError("Must call preprocess() before getting domain")
        return self._domain

    def get_coefficients(self):
        if self._x_grid is None:
            raise ValueError("Must call preprocess() before getting coefficients")
        return {
            'p': self._p_grid.copy(),
            'q': self._q_grid.copy(),
            'w': self._w_grid.copy()
        }
