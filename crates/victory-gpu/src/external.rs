//! Host work that a host library records into a title's command list in
//! place of a console one (the host's FidelityFX upscaler, say): the host
//! device, a title texture's host copy in the state the library is told,
//! and the host list itself.

use windows::Win32::Graphics::Direct3D12::{
    ID3D12Device, ID3D12GraphicsCommandList, ID3D12Resource,
};

use crate::object::Object;

/// The host D3D12 device, created on first use; `None` without a host GPU.
pub fn host_device() -> Option<ID3D12Device> {
    crate::host::host().map(|h| h.device.clone())
}

/// A title texture's host copy, refreshed and moved to the host resource
/// `state` in the title list `list`, and the state it is in: `state` without
/// the bits the host copy cannot take (see `host::legal_state`). What the
/// list records next finds it there and must leave it there. Requesting
/// `COPY_DEST` declares a host write when this list is submitted. The list
/// holds the texture until it is reset. `None` for a buffer, or without a
/// host list.
pub fn host_texture(
    list: *const Object,
    resource: *const Object,
    state: u32,
) -> Option<(ID3D12Resource, u32)> {
    let host = crate::list::host_list(list)?;
    // One `HostTexture` held across refresh and state tracking so concurrent
    // eviction cannot replace what the list tracks behind what is returned.
    let copy = crate::resource::host_copy(resource)?;
    host_texture_for(host, resource, &copy, state)
}

fn host_texture_for(
    host: &crate::host::HostList,
    resource: *const Object,
    copy: &std::sync::Arc<crate::resource::HostTexture>,
    state: u32,
) -> Option<(ID3D12Resource, u32)> {
    const READ: u32 = 0x40 | 0x80 | 0x800; // shader resource, copy source
    const TARGET: u32 = 0x4 | 0x8 | 0x10; // render target, unordered access, depth write
    const COPY_DEST: u32 = 0x400;
    copy.stamp();
    if state & READ != 0 && state & TARGET == 0 {
        if let Some(from) = crate::resource::alias_source_for(resource, copy) {
            crate::texture::alias_copy_to(host, from, resource, copy);
        }
        if crate::texture::title_written(&copy.texture) {
            crate::texture::upload_copy(host, resource, copy);
        }
    }
    #[cfg(feature = "diagnostics")]
    if state & TARGET != 0 {
        crate::capture::note_target(host, resource);
    }
    if let Some(Some(twin)) = copy.twin.get() {
        const UNORDERED_ACCESS: u32 = 0x8;
        if state & UNORDERED_ACCESS != 0 && twin.planes.len() != 1 {
            return None;
        }
        if crate::twin::route(host, copy, twin, state) {
            let plane = &twin.planes[0];
            if let Some(barrier) = host.move_texture(plane, state) {
                // SAFETY: an open list and a live texture.
                unsafe { host.list.ResourceBarrier(&[barrier]) };
            }
            let state = crate::host::legal_state(&plane.texture, state);
            return Some((plane.texture.clone(), state));
        }
    }
    if let Some(barrier) = host.move_texture(copy, state) {
        // SAFETY: an open list and a live texture.
        unsafe { host.list.ResourceBarrier(&[barrier]) };
    }
    let state = crate::host::legal_state(&copy.texture, state);
    if state == COPY_DEST && crate::texture::title_written(&copy.texture) {
        crate::texture::host_writes_copy(host, copy);
    }
    Some((copy.texture.clone(), state))
}

/// Runs `f` with the host list behind the title list `list` (open for
/// recording); `None` without one. What `f` binds (descriptor heaps, root
/// signatures, pipeline) is replaced before the list's next draw or
/// dispatch.
pub fn record<R>(
    list: *const Object,
    f: impl FnOnce(&ID3D12GraphicsCommandList) -> R,
) -> Option<R> {
    let mut out = None;
    crate::list::with_state(list, |host, state| {
        out = Some(f(&host.list));
        state.host_state_lost();
    });
    out
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering::Relaxed;

    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_D32_FLOAT, DXGI_SAMPLE_DESC};
    use windows::core::Interface;

    use super::*;
    use crate::object::{S_OK, This, guid, slot};

    const SHADER_RESOURCE: u32 = 0x40 | 0x80;
    const UNORDERED_ACCESS: u32 = 0x8;
    const DEPTH_WRITE: u32 = 0x10;
    const COPY_DEST: u32 = 0x400;
    const COPY_SOURCE: u32 = 0x800;

    fn texture_words(width: u32, height: u32, format: u32, layout: u32, flags: u32) -> [u32; 14] {
        [
            3,
            0,
            0,
            0,
            width,
            0,
            height,
            1 | 1 << 16,
            format,
            1,
            0,
            layout,
            flags,
            0,
        ]
    }

    fn texture_at(address: u64, words: &[u32; 14]) -> This {
        let iid = guid(
            0x696442be,
            0xa72e,
            0x4059,
            [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
        );
        let mut out = std::ptr::null();
        assert_eq!(
            crate::resource::create_at_address(
                std::ptr::null(),
                address,
                words.as_ptr().cast(),
                0,
                std::ptr::null(),
                &iid,
                &mut out,
            ),
            S_OK
        );
        out
    }

    fn new_title_list() -> (This, This) {
        let null = std::ptr::null();
        let alloc_iid = guid(
            0x6102dee4,
            0xaf59,
            0x4b09,
            [0xb9, 0x99, 0xb4, 0x4d, 0x73, 0xf0, 0x9b, 0x24],
        );
        let list_iid = guid(
            0x5b160d0f,
            0xac1b,
            0x4185,
            [0x8b, 0xa8, 0xb3, 0xae, 0x42, 0xa5, 0xa4, 0x55],
        );
        let mut allocator = null;
        assert_eq!(
            crate::list::create_allocator(null, 0, &alloc_iid, &mut allocator),
            S_OK
        );
        let mut list = null;
        assert_eq!(
            crate::list::create_list(null, 0, 0, allocator, null, &list_iid, &mut list),
            S_OK
        );
        (allocator, list)
    }

    fn close_list(list: This) {
        assert_eq!(slot::<extern "system" fn(This) -> i32>(list, 0x48)(list), S_OK);
    }

    fn reset_list(list: This, allocator: This) {
        assert_eq!(
            slot::<extern "system" fn(This, This, This) -> i32>(list, 0x50)(
                list,
                allocator,
                std::ptr::null(),
            ),
            S_OK
        );
    }

    fn release(o: This) -> u32 {
        slot::<extern "system" fn(This) -> u32>(o, 0x10)(o)
    }

    fn readback_buffer(device: &ID3D12Device, size: u64) -> ID3D12Resource {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(256),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut out: Option<ID3D12Resource> = None;
        // SAFETY: plain readback buffer creation.
        unsafe {
            device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut out,
                )
                .unwrap();
        }
        out.unwrap()
    }

    fn readback_value<T: Copy>(buffer: &ID3D12Resource) -> T {
        let mut ptr = std::ptr::null_mut();
        let range = D3D12_RANGE {
            Begin: 0,
            End: size_of::<T>(),
        };
        // SAFETY: GPU work completed before mapping `range`.
        unsafe {
            buffer.Map(0, Some(&range), Some(&mut ptr)).unwrap();
            let value = ptr.cast::<T>().read_unaligned();
            buffer.Unmap(0, None);
            value
        }
    }

    fn copy_locations(
        buffer: &ID3D12Resource,
        fp: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
        texture: &ID3D12Resource,
    ) -> (D3D12_TEXTURE_COPY_LOCATION, D3D12_TEXTURE_COPY_LOCATION) {
        let placed = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: borrowed for `CopyTextureRegion`.
            pResource: unsafe { std::mem::transmute_copy(buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: fp },
        };
        let sub = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
        };
        (placed, sub)
    }

    /// A texture filled in title memory (no render-target, depth or unordered
    /// flag, such as a 1x1 exposure or mask texture passed to a host upscaler)
    /// uploads its initial and updated bytes before external host work reads it.
    #[test]
    fn title_written_textures_upload_before_external_reads() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let pool = crate::watch::tests::pool(2);
        // Linear 4x4 R8G8B8A8_UNORM (`layout = 1`, 256-byte row pitch on console).
        let tex = texture_at(pool as u64, &texture_words(4, 4, 28, 1, 0));
        // SAFETY: `pool` holds two committed 64 KiB pages.
        unsafe { (pool as *mut u32).write_volatile(0xff33_2211) };

        let (allocator, list) = new_title_list();
        let hl = crate::list::host_list(list).unwrap();
        let (host_tex, state) = host_texture(list, tex, SHADER_RESOURCE).unwrap();
        assert_eq!(state, SHADER_RESOURCE);

        let mut fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut total = 0u64;
        // SAFETY: live host texture and sized output.
        unsafe {
            host.device.GetCopyableFootprints(
                &host_tex.GetDesc(),
                0,
                1,
                0,
                Some(&mut fp),
                None,
                None,
                Some(&mut total),
            );
        }
        let readback = readback_buffer(&host.device, total);

        for want in [0xff33_2211u32, 0xffcc_bbaa] {
            if want != 0xff33_2211 {
                // SAFETY: `pool` is committed; guarded pages fault into the watch handler.
                unsafe { (pool as *mut u32).write_volatile(want) };
                let (_, s) = host_texture(list, tex, SHADER_RESOURCE).unwrap();
                assert_eq!(s, SHADER_RESOURCE);
            }
            let (t, s) = host_texture(list, tex, COPY_SOURCE).unwrap();
            assert_eq!(s, COPY_SOURCE);
            let (dst, src) = copy_locations(&readback, fp, &t);
            // SAFETY: open list; `t` is in `COPY_SOURCE` and `readback` is writable.
            record(list, |l| unsafe {
                l.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
            })
            .unwrap();
            close_list(list);
            host.execute(&[hl]);
            assert_eq!(host.log_messages(), None);
            assert_eq!(readback_value::<u32>(&readback), want, "uploaded texel");
            reset_list(list, allocator);
        }
        assert!(crate::texture::uploads(tex) >= 2);

        for o in [tex, list, allocator] {
            assert_eq!(release(o), 0);
        }
        crate::watch::tests::free(pool);
    }

    #[test]
    fn external_copy_dest_keeps_host_texels_on_later_submissions() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let pool = crate::watch::tests::pool(1);
        let tex = texture_at(pool as u64, &texture_words(4, 4, 28, 1, 0));
        let title_texel = 0xff33_2211u32;
        let host_texel = 0xffcc_bbaau32;
        // SAFETY: the first texel of a committed page.
        unsafe { (pool as *mut u32).write_volatile(title_texel) };

        let (reader_allocator, reader) = new_title_list();
        let (texture, state) = host_texture(reader, tex, COPY_SOURCE).unwrap();
        assert_eq!(state, COPY_SOURCE);
        let mut fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut total = 0;
        // SAFETY: live host texture and sized outputs.
        unsafe {
            host.device.GetCopyableFootprints(
                &texture.GetDesc(),
                0,
                1,
                0,
                Some(&mut fp),
                None,
                None,
                Some(&mut total),
            );
        }
        let readback = readback_buffer(&host.device, total);
        let (dst, src) = copy_locations(&readback, fp, &texture);
        // SAFETY: the list is open and both resources are in their copy states.
        record(reader, |l| unsafe { l.CopyTextureRegion(&dst, 0, 0, 0, &src, None) })
            .unwrap();
        close_list(reader);

        let (writer_allocator, writer) = new_title_list();
        let (texture, state) = host_texture(writer, tex, COPY_DEST).unwrap();
        assert_eq!(state, COPY_DEST);
        let (upload, mapped) = host.upload_buffer(total).unwrap();
        // SAFETY: the upload buffer contains the first texel of the footprint.
        unsafe { mapped.cast::<u32>().write(host_texel) };
        let (src, dst) = copy_locations(&upload, fp, &texture);
        let one_texel = D3D12_BOX { right: 1, bottom: 1, back: 1, ..Default::default() };
        // SAFETY: the list is open and both resources are in their copy states.
        record(writer, |l| unsafe {
            l.CopyTextureRegion(&dst, 0, 0, 0, &src, Some(&one_texel))
        })
        .unwrap();
        close_list(writer);

        let reading = crate::list::host_list(reader).unwrap();
        host.execute(&[reading]);
        assert_eq!(readback_value::<u32>(&readback), title_texel);

        // SAFETY: the first texel of a committed page.
        unsafe { (pool as *mut u32).write_volatile(0xff66_5544) };
        host.execute(&[crate::list::host_list(writer).unwrap()]);
        host.execute(&[reading]);
        assert_eq!(host.log_messages(), None);
        assert_eq!(readback_value::<u32>(&readback), host_texel);

        for o in [reader, writer, reader_allocator, writer_allocator, tex] {
            assert_eq!(release(o), 0);
        }
        crate::watch::tests::free(pool);
    }

    /// A placed texture aliasing GPU-written memory in an incompatible format
    /// family copies from the writer's host texture on its very first external
    /// read, without needing an earlier descriptor view.
    #[test]
    fn aliased_textures_copy_from_their_gpu_writer_on_first_external_read() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let pool = crate::watch::tests::pool(16);
        // 16x16 R32G32_UINT (UAV) and 64x64 BC1_UNORM share 16x16 8-byte blocks.
        let writer = texture_at(pool as u64, &texture_words(16, 16, 17, 0, 4));
        let placed = texture_at(pool as u64, &texture_words(64, 64, 71, 0, 0));

        let writer_copy = crate::resource::host_copy(writer).unwrap();
        let mut wfp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut rows = 0u32;
        let mut total = 0u64;
        // SAFETY: live host texture and sized outputs.
        unsafe {
            host.device.GetCopyableFootprints(
                &writer_copy.texture.GetDesc(),
                0,
                1,
                0,
                Some(&mut wfp),
                Some(&mut rows),
                None,
                Some(&mut total),
            );
        }
        let (upload, up_ptr) = host.upload_buffer(total).unwrap();
        for r in 0..rows as usize {
            for x in 0..16usize {
                let at = r * wfp.Footprint.RowPitch as usize + x * 8;
                // SAFETY: within `upload`'s mapped footprint.
                unsafe {
                    up_ptr.add(at).cast::<u32>().write(0xf800_f800);
                    up_ptr.add(at + 4).cast::<u32>().write(0);
                }
            }
        }

        let (allocator, list) = new_title_list();
        let hl = crate::list::host_list(list).unwrap();
        let (w_tex, s) = host_texture(list, writer, COPY_DEST).unwrap();
        assert_eq!(s, COPY_DEST);
        let (up_loc, w_sub) = copy_locations(&upload, wfp, &w_tex);
        // SAFETY: open list; `w_tex` is in `COPY_DEST`.
        record(list, |l| unsafe {
            l.CopyTextureRegion(&w_sub, 0, 0, 0, &up_loc, None);
        })
        .unwrap();

        let before_aliases = crate::texture::ALIAS_COPIES.load(Relaxed);
        let (_, s) = host_texture(list, placed, SHADER_RESOURCE).unwrap();
        assert_eq!(s, SHADER_RESOURCE);
        assert_eq!(
            crate::texture::ALIAS_COPIES.load(Relaxed),
            before_aliases + 1,
            "first external read copies from the incompatible GPU writer"
        );

        let (p_tex, s) = host_texture(list, placed, COPY_SOURCE).unwrap();
        assert_eq!(s, COPY_SOURCE);
        let mut pfp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut ptotal = 0u64;
        // SAFETY: live host texture and sized outputs.
        unsafe {
            host.device.GetCopyableFootprints(
                &p_tex.GetDesc(),
                0,
                1,
                0,
                Some(&mut pfp),
                None,
                None,
                Some(&mut ptotal),
            );
        }
        let readback = readback_buffer(&host.device, ptotal);
        let (rb_loc, p_sub) = copy_locations(&readback, pfp, &p_tex);
        // SAFETY: open list; `p_tex` is in `COPY_SOURCE`.
        record(list, |l| unsafe {
            l.CopyTextureRegion(&rb_loc, 0, 0, 0, &p_sub, None);
        })
        .unwrap();
        close_list(list);
        host.execute(&[hl]);
        assert_eq!(host.log_messages(), None);
        assert_eq!(readback_value::<[u32; 2]>(&readback), [0xf800_f800, 0]);

        reset_list(list, allocator);
        drop(writer_copy);
        for o in [placed, writer, list, allocator] {
            assert_eq!(release(o), 0);
        }
        crate::watch::tests::free(pool);
    }

    /// `host_texture` routes depth textures with writable twins: unordered
    /// access returns a single-plane twin's colour texture in `UNORDERED_ACCESS`,
    /// and a shader read copies newer twin writes into the depth texture first.
    #[test]
    fn depth_twins_route_unordered_and_shader_reads_for_external_callers() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let pool = crate::watch::tests::pool(4);
        // Single-plane `D32_FLOAT` (`40`) with `DEPTH | UAV` (`6`).
        let depth = texture_at(pool as u64, &texture_words(8, 8, 40, 0, 6));
        let depth_copy = crate::resource::host_copy(depth).unwrap();
        let uav_desc = [41u32, 4, 0, 0, 0, 0, 0, 0, 0, 0];
        let twin_plane = crate::twin::uav(depth, &depth_copy, Some(&uav_desc), host.cpu_view())
            .expect("single-plane depth twin");

        let (allocator, list) = new_title_list();
        let hl = crate::list::host_list(list).unwrap();

        // Clear depth to 0.25, then request `UNORDERED_ACCESS` externally:
        // `twin::route` copies `DEPTH -> TWIN` and returns `twin_plane` in 0x8.
        let (d_tex, s) = host_texture(list, depth, DEPTH_WRITE).unwrap();
        assert_eq!(s, DEPTH_WRITE);
        let dsv_desc = D3D12_DEPTH_STENCIL_VIEW_DESC {
            Format: DXGI_FORMAT_D32_FLOAT,
            ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2D,
            ..Default::default()
        };
        let dsv = host.dsv(&d_tex, Some(&dsv_desc));
        // SAFETY: open list; `d_tex` is in `DEPTH_WRITE`.
        record(list, |l| unsafe {
            l.ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 0.25, 0, None);
        })
        .unwrap();

        let (uav_tex, s) = host_texture(list, depth, UNORDERED_ACCESS).unwrap();
        assert_eq!(s, UNORDERED_ACCESS);
        assert_eq!(uav_tex.as_raw(), twin_plane.as_raw());

        // Reading depth back as `SHADER_RESOURCE` then `COPY_SOURCE` copies
        // `TWIN -> DEPTH` and returns the depth resource itself.
        let (sr_tex, s) = host_texture(list, depth, SHADER_RESOURCE).unwrap();
        assert_eq!(s, SHADER_RESOURCE);
        assert_eq!(sr_tex.as_raw(), depth_copy.texture.as_raw());

        let (cp_tex, s) = host_texture(list, depth, COPY_SOURCE).unwrap();
        assert_eq!(s, COPY_SOURCE);
        let mut fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut total = 0u64;
        // SAFETY: live depth resource and sized output.
        unsafe {
            host.device.GetCopyableFootprints(
                &cp_tex.GetDesc(),
                0,
                1,
                0,
                Some(&mut fp),
                None,
                None,
                Some(&mut total),
            );
        }
        let readback = readback_buffer(&host.device, total);
        let (dst, src) = copy_locations(&readback, fp, &cp_tex);
        // SAFETY: open list; `cp_tex` is in `COPY_SOURCE`.
        record(list, |l| unsafe {
            l.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
        })
        .unwrap();
        close_list(list);
        host.execute(&[hl]);
        assert_eq!(host.log_messages(), None);
        let sample = readback_value::<f32>(&readback);
        assert!((sample - 0.25).abs() < 1e-6, "got {sample}");

        // A two-plane depth+stencil twin cannot hand out one UAV resource,
        // without disturbing shader reads of the depth resource itself.
        let ds = texture_at((pool as u64) + 0x2_0000, &texture_words(8, 8, 20, 0, 6));
        let ds_copy = crate::resource::host_copy(ds).unwrap();
        let _ = crate::twin::uav(ds, &ds_copy, Some(&uav_desc), host.cpu_view())
            .expect("two-plane depth+stencil twin");
        reset_list(list, allocator);
        assert!(host_texture(list, ds, UNORDERED_ACCESS).is_none());
        let (ds_sr, ds_state) = host_texture(list, ds, SHADER_RESOURCE).unwrap();
        assert_eq!(ds_state, SHADER_RESOURCE);
        assert_eq!(ds_sr.as_raw(), ds_copy.texture.as_raw());
        close_list(list);
        host.execute(&[hl]);
        assert_eq!(host.log_messages(), None);

        reset_list(list, allocator);
        drop((depth_copy, ds_copy));
        for o in [ds, depth, list, allocator] {
            assert_eq!(release(o), 0);
        }
        crate::watch::tests::free(pool);
    }

    /// If eviction clears `placed`'s cached slot (or another use replaces the
    /// memory's shared texture) after `host_texture` captures `copy`, alias
    /// lookup, alias copy, and title-memory upload still refresh and return
    /// that exact `copy` rather than populating a replacement behind it.
    #[test]
    fn eviction_after_capturing_copy_refreshes_the_held_host_texture() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let pool = crate::watch::tests::pool(16);

        // 1. Alias path: capture `placed_copy`, then evict `placed`'s cached
        //    slot (`made_copy` becomes `None`) before `host_texture_for` runs.
        let writer = texture_at(pool as u64, &texture_words(16, 16, 17, 0, 4));
        let placed = texture_at(pool as u64, &texture_words(64, 64, 71, 0, 0));
        let writer_copy = crate::resource::host_copy(writer).unwrap();
        let placed_copy = crate::resource::host_copy(placed).unwrap();
        assert!(crate::resource::give_up_now(placed));
        assert!(crate::resource::made_copy(placed).is_none());

        let mut wfp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut rows = 0u32;
        let mut total = 0u64;
        // SAFETY: live host texture and sized outputs.
        unsafe {
            host.device.GetCopyableFootprints(
                &writer_copy.texture.GetDesc(),
                0,
                1,
                0,
                Some(&mut wfp),
                Some(&mut rows),
                None,
                Some(&mut total),
            );
        }
        let (upload, up_ptr) = host.upload_buffer(total).unwrap();
        for r in 0..rows as usize {
            for x in 0..16usize {
                let at = r * wfp.Footprint.RowPitch as usize + x * 8;
                // SAFETY: within `upload`'s mapped footprint.
                unsafe {
                    up_ptr.add(at).cast::<u32>().write(0x07e0_07e0);
                    up_ptr.add(at + 4).cast::<u32>().write(0xffff_ffff);
                }
            }
        }

        let (allocator, list) = new_title_list();
        let hl = crate::list::host_list(list).unwrap();
        let (w_tex, _) = host_texture(list, writer, COPY_DEST).unwrap();
        let (up_loc, w_sub) = copy_locations(&upload, wfp, &w_tex);
        // SAFETY: open list; `w_tex` is in `COPY_DEST`.
        record(list, |l| unsafe {
            l.CopyTextureRegion(&w_sub, 0, 0, 0, &up_loc, None);
        })
        .unwrap();

        let before_aliases = crate::texture::ALIAS_COPIES.load(Relaxed);
        let (got_sr, s) = host_texture_for(hl, placed, &placed_copy, SHADER_RESOURCE).unwrap();
        assert_eq!(s, SHADER_RESOURCE);
        assert_eq!(got_sr.as_raw(), placed_copy.texture.as_raw());
        assert_eq!(crate::texture::ALIAS_COPIES.load(Relaxed), before_aliases + 1);
        assert!(
            crate::resource::made_copy(placed).is_none(),
            "refresh used the held copy without re-populating placed's evicted slot"
        );

        let (got_cp, s) = host_texture_for(hl, placed, &placed_copy, COPY_SOURCE).unwrap();
        assert_eq!(s, COPY_SOURCE);
        assert_eq!(got_cp.as_raw(), placed_copy.texture.as_raw());
        let mut pfp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut ptotal = 0u64;
        // SAFETY: live host texture and sized outputs.
        unsafe {
            host.device.GetCopyableFootprints(
                &got_cp.GetDesc(),
                0,
                1,
                0,
                Some(&mut pfp),
                None,
                None,
                Some(&mut ptotal),
            );
        }
        let readback = readback_buffer(&host.device, ptotal);
        let (rb_loc, p_sub) = copy_locations(&readback, pfp, &got_cp);
        // SAFETY: open list; `got_cp` is in `COPY_SOURCE`.
        record(list, |l| unsafe {
            l.CopyTextureRegion(&rb_loc, 0, 0, 0, &p_sub, None);
        })
        .unwrap();
        close_list(list);
        host.execute(&[hl]);
        assert_eq!(readback_value::<[u32; 2]>(&readback), [0x07e0_07e0, 0xffff_ffff]);
        reset_list(list, allocator);

        // 2. Upload path: capture `old_copy` of a read-only texture, evict its
        //    cached slot (`made_copy` becomes `None`), and verify `upload_copy`
        //    and `staging_for` upload into `old_copy` without touching `tex`'s
        //    evicted slot.
        let up_pool = crate::watch::tests::pool(4);
        let up_addr = up_pool as u64;
        let tex = texture_at(up_addr, &texture_words(4, 4, 28, 1, 0));
        let old_copy = crate::resource::host_copy(tex).unwrap();
        assert!(crate::resource::give_up_now(tex));
        assert!(crate::resource::made_copy(tex).is_none());

        // SAFETY: `up_pool` holds four committed 4 KiB pages.
        unsafe { (up_pool as *mut u32).write_volatile(0xff77_6655) };
        let (up_sr, s) = host_texture_for(hl, tex, &old_copy, SHADER_RESOURCE).unwrap();
        assert_eq!(s, SHADER_RESOURCE);
        assert_eq!(up_sr.as_raw(), old_copy.texture.as_raw());
        assert!(
            crate::resource::made_copy(tex).is_none(),
            "upload_copy used the held copy without re-populating tex's evicted slot"
        );
        let (up_cp, s) = host_texture_for(hl, tex, &old_copy, COPY_SOURCE).unwrap();
        assert_eq!(s, COPY_SOURCE);
        assert_eq!(up_cp.as_raw(), old_copy.texture.as_raw());

        let mut ufp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut utotal = 0u64;
        // SAFETY: live host texture and sized outputs.
        unsafe {
            host.device.GetCopyableFootprints(
                &up_cp.GetDesc(),
                0,
                1,
                0,
                Some(&mut ufp),
                None,
                None,
                Some(&mut utotal),
            );
        }
        let ureadback = readback_buffer(&host.device, utotal);
        let (urb_loc, u_sub) = copy_locations(&ureadback, ufp, &up_cp);
        // SAFETY: open list; `up_cp` is in `COPY_SOURCE`.
        record(list, |l| unsafe {
            l.CopyTextureRegion(&urb_loc, 0, 0, 0, &u_sub, None);
        })
        .unwrap();
        close_list(list);
        host.execute(&[hl]);
        assert_eq!(readback_value::<u32>(&ureadback), 0xff77_6655);

        reset_list(list, allocator);
        drop((writer_copy, placed_copy, old_copy));
        for o in [tex, placed, writer, list, allocator] {
            assert_eq!(release(o), 0);
        }
        crate::watch::tests::free(up_pool);
        crate::watch::tests::free(pool);
    }
}
