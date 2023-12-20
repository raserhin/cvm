use cosmwasm_std::{Deps, Order, StdResult};
use cvm_route::asset::{AssetItem, AssetReference};
use cvm_runtime::{AssetId, NetworkId};

use crate::prelude::*;

pub type NetworkAssetId = (NetworkId, AssetId);

/// when assets to be sent to other network it should be mapped before sent
pub(crate) const NETWORK_ASSET: Map<NetworkAssetId, AssetId> = Map::new("network_asset");

pub(crate) const ASSETS: Map<AssetId, AssetItem> = Map::new("assets");
pub(crate) const LOCAL_ASSETS: Map<AssetReference, AssetItem> = Map::new("local_assets");

pub fn get_all_assets(deps: Deps) -> StdResult<Vec<AssetItem>> {
    ASSETS
        .range(deps.storage, None, None, Order::Ascending)
        .map(|x| x.map(|(_, x)| x))
        .collect()
}

pub fn get_all_network_assets(deps: Deps) -> StdResult<Vec<(NetworkAssetId, AssetId)>> {
    NETWORK_ASSET
        .range(deps.storage, None, None, Order::Ascending)
        .map(|x| x.map(|(((network_id, asset_id)), x)| ((network_id.into(), asset_id.into()), x)))
        .collect()
}
