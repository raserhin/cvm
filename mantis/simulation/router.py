# solves using convex optimization
import numpy as np
import cvxpy as cp
from abc import ABC, abstractmethod, abstractproperty

from simulation.data import TAssetId, TNetworkId


# simulate denom paths to and from chains, with center node
def populate_chain_dict(
    chains: dict[TNetworkId, list[TAssetId]], center_node: TNetworkId
):
    # Add tokens with denom to Center Node
    # Basic IBC transfer
    for chain, tokens in chains.items():
        if chain != center_node:
            chains[center_node].extend(f"{chain}/{token}" for token in tokens)

    1  # Add tokens from Center Node to outers

    # Simulate IBC transfer through Composable Cosmos
    for chain, tokens in chains.items():
        if chain != center_node:
            chains[chain].extend(
                f"{center_node}/{token}"
                for token in chains[center_node]
                if f"{chain}/" not in token
            )


def solve(
    all_tokens: list[TAssetId],
    all_cfmms: list[tuple[TAssetId, TAssetId]],
    reserves: list[np.ndarray[np.float64]],
    cfmm_tx_cost: list[float],
    fees: list[float],
    ibc_pools: int,
    origin_token: TAssetId,
    number_of_init_tokens: float,
    obj_token: TAssetId,
    force_eta: list[float] = None,
):
    # Build local-global matrices
    count_tokens = len(all_tokens)
    count_cfmms = len(all_cfmms)

    current_assets = np.zeros(count_tokens)  # Initial assets
    current_assets[all_tokens.index(origin_token)] = number_of_init_tokens

    A = []
    for cfmm in all_cfmms:
        n_i = len(cfmm)  # Number of tokens in pool (default to 2)
        A_i = np.zeros((count_tokens, n_i))
        for i, token in enumerate(cfmm):
            A_i[all_tokens.index(token), i] = 1
        A.append(A_i)

    # Build variables

    # tendered (given) amount
    deltas = [cp.Variable(len(l), nonneg=True) for l in all_cfmms]

    # received (wanted) amounts
    lambdas = [cp.Variable(len(l), nonneg=True) for l in all_cfmms]
    
    eta = cp.Variable(
        count_cfmms, boolean=True
    )  # Binary value, indicates tx or not for given pool

    # network trade vector - net amount received over all trades(transfers/exchanges)
    psi = cp.sum(
        [A_i @ (LAMBDA - DELTA) for A_i, DELTA, LAMBDA in zip(A, deltas, lambdas)]
    )

    # Objective is to trade number_of_init_tokens of asset origin_token for a maximum amount of asset objective_token
    obj = cp.Maximize(psi[all_tokens.index(obj_token)] - cp.sum(eta @ cfmm_tx_cost))

    # Reserves after trade
    new_reserves = [
        R + gamma_i * D - L for R, gamma_i, D, L in zip(reserves, fees, deltas, lambdas)
    ]

    # Trading function constraints
    constrains = [
        psi + current_assets >= 0,
    ]

    # Pool constraint (Uniswap v2 like)
    for i in range(count_cfmms - ibc_pools):
        constrains.append(cp.geo_mean(new_reserves[i]) >= cp.geo_mean(reserves[i]))

    # Pool constraint for IBC transfer (constant sum)
    # NOTE: Ibc pools are at the bottom of the cfmm list
    for i in range(count_cfmms - ibc_pools, count_cfmms):
        constrains.append(cp.sum(new_reserves[i]) >= cp.sum(reserves[i]))
        constrains.append(new_reserves[i] >= 0)

    # Enforce deltas depending on pass or not pass variable
    # MAX_RESERVE should be big enough so delta <<< MAX_RESERVE
    for i in range(count_cfmms):
        constrains.append(deltas[i] <= eta[i] * reserves[i] * 0.2)
        constrains.append(eta[i] <= 1)
        # constrains.append(d == eta[i])

        if force_eta:
            constrains.append(eta[i] == force_eta[i])

    # Set up and solve problem
    prob = cp.Problem(obj, constrains)
    # success: CLARABEL,
    # failed: ECOS, GLPK, GLPK_MI, CVXOPT, SCIPY, CBC, SCS
    # 
    # GLOP, SDPA, GUROBI, OSQP, CPLEX, MOSEK, , COPT, XPRESS, PIQP, PROXQP, NAG, PDLP, SCIP, DAQP
    prob.solve(solver=cp.MOSEK, verbose=True)

    print(
        f"\033[1;91m][{count_cfmms}]Total amount out: {psi.value[all_tokens.index(obj_token)]}\033[0m"
    )

    for i in range(count_cfmms):
        print(
            f"Market {all_cfmms[i][0]}<->{all_cfmms[i][1]}, delta: {deltas[i].value}, lambda: {lambdas[i].value}, eta: {eta[i].value}",
        )

    # deltas[i] - how much one gives to pool i
    # lambdas[i] - how much one wants to get from pool i
    return deltas, lambdas, psi, eta


from simulation.data import AllData, Input
from cvxpy.expressions.expression import Expression
from cvxpy.expressions.variable import Variable
from dataclasses import dataclass


@dataclass
class RouterAsset:
    id: int
    price: float


@dataclass
class RouterExchange(ABC):
    assets: list[RouterAsset]
    fix_cost: float
    reserves: list[float]
    fee: float

    @abstractmethod
    def constraints(self, new_reserve: Expression) -> list[Expression]:
        pass


class ConstantProductPool(RouterExchange):
    def constraints(self, new_reserve: Expression) -> list[Expression]:
        return [cp.geo_mean(new_reserve) >= cp.geo_mean(self.reserves)]


class IbcTransfer(RouterExchange):
    def constraints(self, new_reserve: Expression) -> list[Expression]:
        return [cp.sum(new_reserve) >= cp.sum(self.reserves), new_reserve >= 0]


class Router(ABC):
    @abstractmethod
    def solve():
        pass


class ConvexRouter(Router):
    def get_new_reserve(
        self, delta: Variable, lamb: Variable, exchange: RouterExchange
    ) -> Expression:
        return exchange.reserves + (1 - exchange.fee) * delta - lamb

    def solve(
        self, exchanges: list[RouterExchange], assets: list[RouterAsset], input: Input
    ):
        asset_index = lambda token_id: [
            i for i, asset in enumerate(assets) if asset.id == token_id
        ][0]
        is_micp = len(exchanges) <= 20
        num_exchanges = len(exchanges)
        num_assets = len(assets)

        initial_assets = np.zeros(num_assets)  # Initial assets

        initial_assets[asset_index(input.in_token_id)] = input.in_amount

        A = []
        for exchange in exchanges:
            n_i = len(exchange.assets)  # Number of tokens in pool (default to 2)
            A_i = np.zeros((num_assets, n_i))
            for i, token in enumerate(exchange.assets):
                A_i[asset_index(token.id), i] = 1
            A.append(A_i)

        # tendered (given) amount
        deltas = [
            cp.Variable(len(exchange.assets), nonneg=True) for exchange in exchanges
        ]

        # received (wanted) amounts
        lambdas = [
            cp.Variable(len(exchange.assets), nonneg=True) for exchange in exchanges
        ]

        eta = cp.Variable(
            num_exchanges, boolean=is_micp
        )  # Binary value, indicates tx or not for given pool

        # network trade vector - net amount received over all trades(transfers/exchanges)
        psi = cp.sum(
            [A_i @ (LAMBDA - DELTA) for A_i, DELTA, LAMBDA in zip(A, deltas, lambdas)]
        )

        # Objective is to trade number_of_init_tokens of asset origin_token for a maximum amount of asset objective_token
        obj = cp.Maximize(
            psi[asset_index(input.out_token_id)]
            - cp.sum(eta @ [exchange.fix_cost for exchange in exchanges])
        )

        # Trading function constraints
        constrains = [
            psi + initial_assets >= 0,
        ]

        for i, exchange in enumerate(exchanges):
            new_reserve = self.get_new_reserve(deltas[i], lambdas[i], exchange)
            constrains.extend(exchange.constraints(new_reserve))
            if not is_micp:
                # constrains.append(eta <= 1)
                constrains.append(
                    cp.sum(deltas[i] @ [asset.price for asset in exchange.assets])
                    <= eta[i]
                    * 60
                    * input.in_amount
                    * assets[asset_index(input.in_token_id)].price
                )
            else:
                constrains.append(deltas[i] <= eta[i] * exchange.reserves * 0.5)

        # Set up and solve problem
        prob = cp.Problem(obj, constrains)
        prob.solve(solver=cp.MOSEK)

        print(
            f"\033[1;91m[{num_exchanges}]Total amount out: {psi.value[asset_index(input.out_token_id)]}\033[0m"
        )

        for i in range(num_exchanges):
            print(
                f"Market {exchanges[i].assets[0]}<->{exchanges[i].assets[1]}, delta: {deltas[i].value}, lambda: {lambdas[i].value}, eta: {eta[i].value}",
            )
        if not is_micp:
            exchanges = [
                pair[0]
                for pair in sorted(
                    zip(exchanges, eta), key=lambda ele: ele[1].value, reverse=True
                )
            ][:20]
            self.solve(exchanges, assets, input)
        else:
            return deltas, lambdas, psi, eta


if __name__ == "__main__":
    import random

    assets = []
    for i in range(11):
        assets.append(RouterAsset(i + 1, random.randint(8, 15)))

    exchanges = []
    for i in range(50):
        if random.randint(0, 1):
            exchanges.append(
                IbcTransfer(
                    [assets[random.randint(0, 10)], assets[random.randint(0, 10)]],
                    1,
                    [random.randint(1000, 1100), random.randint(1000, 1400)],
                    0.03,
                )
            )
        else:
            exchanges.append(
                ConstantProductPool(
                    [assets[random.randint(0, 10)], assets[random.randint(0, 10)]],
                    1,
                    [random.randint(1000, 1100), random.randint(1000, 1400)],
                    0.02,
                )
            )
    router = ConvexRouter()

    input_obj = Input(
        in_token_id=1, out_token_id=4, in_amount=1, out_amount=100, max=False
    )
    router.solve(exchanges, assets, input_obj)
