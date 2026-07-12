"""
dds_ddss.py

Thin ctypes wrapper around the ddss fork's CalcAllTablesPBNx API.
Provides a drop-in replacement for endplay's calc_all_tables that supports
arbitrary batch sizes (the library chunks internally at 1000 deals).

If the ddss DLL is not found, DDSS_AVAILABLE is set to False and the module
can be imported without error -- callers should check DDSS_AVAILABLE before
calling any functions.
"""

import ctypes
import pathlib
import sys
from typing import List, Optional

import endplay._dds as endplay_dds

# TODO: how can ddss dll be made available? Does it ddss need to be made available as a package like endplay?
_DLL_PATH = pathlib.Path(r"C:\sw\bridge\ML-Contract-Bridge\src\ddss\build-cmake\Release\dds.dll")

DDSS_AVAILABLE = False
_dll = None


class ddTableDealPBN(ctypes.Structure):
    _fields_ = [("cards", ctypes.c_char * 80)]


class ddTableResults(ctypes.Structure):
    _fields_ = [("resTable", (ctypes.c_int * 4) * 5)]  # [strain][hand]


class parResults(ctypes.Structure):
    _fields_ = [
        ("parScore", (ctypes.c_char * 16) * 2),
        ("parContractsString", (ctypes.c_char * 128) * 2),
    ]


class DDSSResult:
    """Mimics endplay's DDTable interface so downstream code works unchanged.

    resTable layout: [strain][hand] where
        strain: 0=S, 1=H, 2=D, 3=C, 4=N
        hand:   0=N, 1=E, 2=S, 3=W
    """

    def __init__(self, res: ddTableResults):
        self._res = res
        # Some endplay helpers such as `par()` reach into DDTable._data directly
        # instead of using only the public indexing API. Mirror the ctypes result
        # into a real endplay ddTableResults object so those callers keep working.
        self._data = endplay_dds.ddTableResults()
        for strain in range(5):
            for hand in range(4):
                self._data.resTable[strain][hand] = res.resTable[strain][hand]

    def __getitem__(self, cell):
        strain, hand = cell
        return self._data.resTable[int(strain)][int(hand)]

    def to_list(self, player_major: bool = False) -> list:
        if player_major:
            return [
                [self._data.resTable[d][p] for d in range(5)]
                for p in range(4)
            ]
        return [
            [self._data.resTable[d][p] for p in range(4)]
            for d in range(5)
        ]

    def pprint(self, stream=sys.stdout) -> None:
        """Print the double-dummy table in endplay's grid format."""
        denoms = "CDHSN"
        players = "NSEW"
        print("   ", " ".join(denom.rjust(2) for denom in denoms), file=stream)
        for player in players:
            print(player.rjust(3), end="", file=stream)
            for denom in denoms:
                d = denoms.index(denom)
                p = players.index(player)
                print(str(self._data.resTable[d][p]).rjust(3), end="", file=stream)
            print(file=stream)

    def __str__(self) -> str:
        denoms = "CDHSN"
        players = "NSEW"
        return (
            ",".join(denoms)
            + ";"
            + ";".join(
                player + ":" + ",".join(str(self._data.resTable[d][p]) for d in range(5))
                for p, player in enumerate(players)
            )
        )


def _load_dll() -> bool:
    global _dll, DDSS_AVAILABLE
    if not _DLL_PATH.exists():
        print(f"ddss DLL not found at {_DLL_PATH}. Falling back to endplay DDS.")
        return False
    try:
        if sys.platform == "win32":
            _dll = ctypes.WinDLL(str(_DLL_PATH))
        else:
            _dll = ctypes.CDLL(str(_DLL_PATH))

        _dll.CalcAllTablesPBNx.restype = ctypes.c_int
        _dll.CalcAllTablesPBNx.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ddTableDealPBN),
            ctypes.c_int,
            ctypes.c_int * 5,
            ctypes.POINTER(ddTableResults),
            ctypes.POINTER(parResults),
        ]

        _dll.SetMaxThreads.restype = None
        _dll.SetMaxThreads.argtypes = [ctypes.c_int]
        _dll.SetMaxThreads(0)

        _dll.ErrorMessage.restype = None
        _dll.ErrorMessage.argtypes = [ctypes.c_int, ctypes.c_char_p]

        DDSS_AVAILABLE = True
        print(f"Loaded ddss DLL from {_DLL_PATH}")
        return True
    except OSError as e:
        print(f"Failed to load ddss DLL from {_DLL_PATH}: {e}. Falling back to endplay DDS.")
        return False


_load_dll()


def calc_all_tables_pbnx(
    pbn_strings: List[str],
    mode: int = -1,
    trump_filter: Optional[List[int]] = None,
) -> List[DDSSResult]:
    """Compute double-dummy tables for a list of PBN deal strings.

    Parameters:
        pbn_strings: PBN deal strings (e.g. "N:AK.QJ.T9.8765 ...").
        mode: Par vulnerability mode. -1 = no par, 0-3 = vulnerability.
        trump_filter: 5-element list; 0 = solve strain, 1 = skip.
                      Default solves all strains.

    Returns:
        List of DDSSResult objects (same length as pbn_strings).

    Raises:
        RuntimeError: If DDSS_AVAILABLE is False or the DLL returns an error.
        ValueError: If pbn_strings has negative length (invalid).
    """
    if not DDSS_AVAILABLE:
        raise RuntimeError("ddss DLL is not available")

    n = len(pbn_strings)
    if n == 0:
        return []

    if trump_filter is None:
        trump_filter = [0, 0, 0, 0, 0]

    DealArray = ddTableDealPBN * n
    ResultArray = ddTableResults * n

    deals = DealArray()
    results = ResultArray()
    tf = (ctypes.c_int * 5)(*trump_filter)

    for i, pbn in enumerate(pbn_strings):
        deals[i].cards = pbn.encode("ascii") if isinstance(pbn, str) else pbn

    par_array = None
    par_ptr = None
    if mode >= 0:
        ParArray = parResults * n
        par_array = ParArray()
        par_ptr = ctypes.cast(par_array, ctypes.POINTER(parResults))

    ret = _dll.CalcAllTablesPBNx(
        n,
        deals,
        mode,
        tf,
        results,
        par_ptr,
    )

    if ret != 1:
        buf = ctypes.create_string_buffer(80)
        _dll.ErrorMessage(ret, buf)
        raise RuntimeError(f"ddss CalcAllTablesPBNx error {ret}: {buf.value.decode()}")

    return [DDSSResult(results[i]) for i in range(n)]


def get_par_results(
    pbn_strings: List[str],
    mode: int = 0,
    trump_filter: Optional[List[int]] = None,
) -> List[dict]:
    """Compute DD tables AND par results.

    Parameters:
        pbn_strings: PBN deal strings.
        mode: Vulnerability (0=None, 1=Both, 2=NS, 3=EW).
        trump_filter: 5-element list; 0 = solve strain, 1 = skip.

    Returns:
        List of dicts with 'dd' (DDSSResult) and 'par' (parResults) keys.
    """
    if not DDSS_AVAILABLE:
        raise RuntimeError("ddss DLL is not available")

    n = len(pbn_strings)
    if n == 0:
        return []

    if trump_filter is None:
        trump_filter = [0, 0, 0, 0, 0]

    DealArray = ddTableDealPBN * n
    ResultArray = ddTableResults * n
    ParArray = parResults * n

    deals = DealArray()
    results = ResultArray()
    par = ParArray()
    tf = (ctypes.c_int * 5)(*trump_filter)

    for i, pbn in enumerate(pbn_strings):
        deals[i].cards = pbn.encode("ascii") if isinstance(pbn, str) else pbn

    ret = _dll.CalcAllTablesPBNx(
        n, deals, mode, tf, results,
        ctypes.cast(par, ctypes.POINTER(parResults)),
    )

    if ret != 1:
        buf = ctypes.create_string_buffer(80)
        _dll.ErrorMessage(ret, buf)
        raise RuntimeError(f"ddss CalcAllTablesPBNx error {ret}: {buf.value.decode()}")

    return [
        {
            "dd": DDSSResult(results[i]),
            "par_score": [par[i].parScore[s].value.decode() for s in range(2)],
            "par_contracts": [par[i].parContractsString[s].value.decode() for s in range(2)],
        }
        for i in range(n)
    ]
