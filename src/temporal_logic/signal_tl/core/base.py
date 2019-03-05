from abc import abstractmethod

import numpy as np
import pandas as pd
import sympy
from sympy import sympify
from sympy.core.relational import Ge, Gt, Le, Lt
from sympy.printing.printer import Printer

import math
from abc import ABC, abstractmethod

import sympy


class Signal(sympy.Symbol):
    pass


class Parameter(sympy.Symbol):
    pass


class Expression(ABC):
    """Expression base class

    This class just holds a bunch of methods that determine what kind of expression it is (defined below)
    """

    is_Atom = False
    is_Singleton = False
    is_Predicate = False
    is_LogicOp = False
    is_TemporalOp = False

    nargs = 0

    _params = []

    def __new__(cls, *args, **kwargs):

        _args = cls._filter_args(*args)
        if not (cls.nargs is None or math.isinf(cls.nargs)) and cls.nargs != len(_args):
            raise ValueError('Incompatible number of args')

        obj = object.__new__(cls)
        obj._args = obj._filter_args(*args)
        for a in obj._args:
            a._parent = obj
        obj._mhash = None
        obj._depth = obj._calc_depth()
        obj._size = obj._calc_size()
        obj._parent = None
        return obj

    @classmethod
    def convert(cls, other):
        from temporal_logic.signal_tl.core import basic_ops

        sym_2_expression = {
            'And': basic_ops.And,
            'Or': basic_ops.Or,
            'Not': basic_ops.Not,
            'Implies': basic_ops.Implies,
        }

        if isinstance(other, Expression):
            return other
        if isinstance(other, (bool, sympy.boolalg.BooleanAtom)):
            return true if other else false
        if isinstance(other, (sympy.And, sympy.Or, sympy.Not, sympy.Implies)):
            return sym_2_expression[type(other).__name__](*other.args)
        if isinstance(other, (sympy.Ge, sympy.Gt, sympy.Le, sympy.Lt)):
            return Predicate(other)
        raise TypeError('Incompatible argument type: %s',
                        other.__module__ + "." + other.__class__.__qualname__)

    @classmethod
    def _filter_args(cls, *args) -> tuple:
        return tuple(map(cls.convert, args))

    @property
    def func(self):
        return self.__class__

    @property
    def args(self):
        return self._args

    def __and__(self, other):
        """Logical And"""
        from . import basic_ops
        return basic_ops.And(self, other)

    __rand__ = __and__

    def __or__(self, other):
        """Logical Or"""
        from . import basic_ops
        return basic_ops.Or(self, other)

    __ror__ = __or__

    def __neg__(self):
        """Logical Negation"""
        from . import basic_ops
        return basic_ops.Not(self)

    def __invert__(self):
        """Logical Negation"""
        from . import basic_ops
        return basic_ops.Not(self)

    def __rshift__(self, other):
        """Logical Implication"""
        from . import basic_ops
        return basic_ops.Or(basic_ops.Not(self), other)

    def __lshift__(self, other):
        """Logical Reverse Implication"""
        from . import basic_ops
        return basic_ops.Or(basic_ops.Not(other), self)

    __rrshift__ = __lshift__
    __rlshift__ = __rshift__

    def __hash__(self):
        h = self._mhash
        if h is None:
            h = hash((type(self).__name__,) + self._hashable_content())
            self._mhash = h
        return h

    def _hashable_content(self):
        return self.args

    def _noop(self, other=None):
        raise TypeError('BooleanAtom not allowed in this context.')

    __add__ = _noop
    __radd__ = _noop
    __sub__ = _noop
    __rsub__ = _noop
    __mul__ = _noop
    __rmul__ = _noop
    __pow__ = _noop
    __rpow__ = _noop
    __rdiv__ = _noop
    __truediv__ = _noop
    __div__ = _noop
    __rtruediv__ = _noop
    __mod__ = _noop
    __rmod__ = _noop
    _eval_power = _noop

    @property
    def depth(self):
        return self._depth

    def _calc_depth(self):
        if self.is_Atom:
            return 0
        return 1 + max(map(lambda arg: arg.depth, self.args))

    @property
    def size(self):
        return self._size

    def _calc_size(self):
        if self.is_Atom:
            return 1
        return 1 + sum(map(lambda arg: arg.depth, self.args))

    @property
    def parent(self):
        return self._parent

    @abstractmethod
    def _latex(self, printer: Printer = None):
        pass

    def to_nnf(self):
        return self

    def to_cnf(self):
        return self


class Atom(Expression):
    is_Atom = True
    nargs = 0

    @abstractmethod
    def eval(self, *args):
        pass

    @classmethod
    def _filter_args(cls, *args):
        return ()


class TLTrue(Atom):
    """Atomic True"""
    is_Singleton = True

    def eval(self):
        return True

    def __nonzero__(self):
        return True

    __bool__ = __nonzero__

    def __hash__(self):
        return hash(True)

    def _latex(self, printer=None):
        return r' \top '


class TLFalse(Atom):
    """Atomic False"""
    is_Singleton = True

    def eval(self):
        return False

    def __nonzero__(self):
        return False

    __bool__ = __nonzero__

    def __hash__(self):
        return hash(False)

    def _latex(self, printer=None):
        return r' \bot '


true = TLTrue()
false = TLFalse()


class Predicate(Atom):
    r"""A signal predicate in the form of :math:`f(x_i) \geq 0`

    Define a predicate on a signal in the form of :math:`f(x_i) \geq 0`.
    Here,
        :math:`i \in \mathbb{N}`
        :math:`x_i` are the parameters of the signal.

    """
    _predicate = None
    is_Predicate = True
    _signals = None

    def __new__(cls, *args, **kwargs):
        if len(args) != 1:
            raise ValueError('Must provide the predicate as an argument')
        predicate = cls._get_predicate(args[0])
        obj = super(Predicate, cls).__new__(cls, *args, **kwargs)
        obj._predicate = predicate
        return obj

    @classmethod
    def _get_predicate(cls, args):
        """Return the predicate in the form f(x) >= 0"""
        pred_default = sympify(args)
        if isinstance(pred_default, (Ge, Gt, Le, Lt)):

            new_lhs = pred_default.gts - pred_default.lts

            if isinstance(pred_default, (Ge, Gt)):
                return pred_default.func(new_lhs, 0)
            if isinstance(pred_default, Lt):
                return Ge(new_lhs, 0)
            if isinstance(pred_default, Le):
                return Gt(new_lhs, 0)

        raise TypeError('The given predicate is not an inequality')

    @property
    def expr(self):
        return self._predicate.gts

    @property
    def predicate(self):
        return self._predicate

    @property
    def signals(self):
        if self._signals is None:
            self._signals = set(map(str, self.predicate.atoms(Signal)))
        return self._signals  # type: set

    def f(self, trace):
        """
        Evaluate the RHS of predicate

        Assumption:
            The name of the symbols in this predicate are the same as the name of the columns in the DataFrame.
            If the trace is a series, the number of signals used in the predicate must be equal to 1
        """
        _f = sympy.lambdify(self._signals, self._predicate.gts, 'numpy')

        if isinstance(trace, pd.DataFrame):
            assert self.signals.issubset(
                trace.columns), 'The signals used in the predicates are not a subset of the column names of the trace'
            signals = tuple(trace[i].values for i in self.signals)
            return _f(*signals)

        elif isinstance(trace, pd.Series):
            assert len(
                self.signals) == 1, 'Predicate uses more than 1 symbol, got 1-D trace'
            signal = trace.values
            return _f(signal)
        else:
            raise ValueError(
                'Expected pandas DataFrame or Series, got {}'.format(trace.__qualname__))

    def eval(self, trace):
        """
        Evaluate the predicate.

        :returns: Boolean signal
        """
        if isinstance(self.predicate, sympy.Ge):
            return self.f(trace) >= 0
        return self.f(trace) > 0

    def _latex(self, printer: Printer = None):
        pred = printer.doprint(self.predicate)
        return pred
