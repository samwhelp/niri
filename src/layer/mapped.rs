use niri_config::utils::MergeWith as _;
use niri_config::{Config, LayerRule};
use smithay::backend::renderer::element::surface::WaylandSurfaceRenderElement;
use smithay::backend::renderer::element::Kind;
use smithay::backend::renderer::utils::RendererSurfaceStateUserData;
use smithay::desktop::{LayerSurface, PopupManager};
use smithay::utils::{Logical, Point, Rectangle, Scale, Size};
use smithay::wayland::background_effect::BackgroundEffectSurfaceCachedState;
use smithay::wayland::compositor::{with_states, RectangleKind};
use smithay::wayland::shell::wlr_layer::{ExclusiveZone, Layer};

use super::ResolvedLayerRules;
use crate::animation::Clock;
use crate::layout::shadow::Shadow;
use crate::niri_render_elements;
use crate::protocols::kde_blur::{KdeBlurRegion, KdeBlurSurfaceCachedState};
use crate::render_helpers::background_effect::BackgroundEffectElement;
use crate::render_helpers::blur::BlurElement;
use crate::render_helpers::renderer::NiriRenderer;
use crate::render_helpers::shadow::ShadowRenderElement;
use crate::render_helpers::solid_color::{SolidColorBuffer, SolidColorRenderElement};
use crate::render_helpers::surface::push_elements_from_surface_tree;
use crate::render_helpers::{background_effect, RenderCtx};
use crate::utils::{baba_is_float_offset, round_logical_in_physical};

#[derive(Debug)]
pub struct MappedLayer {
    /// The surface itself.
    surface: LayerSurface,

    /// Up-to-date rules.
    rules: ResolvedLayerRules,

    /// Buffer to draw instead of the surface when it should be blocked out.
    block_out_buffer: SolidColorBuffer,

    /// The shadow around the surface.
    shadow: Shadow,

    blur: BlurElement,

    /// The view size for the layer surface's output.
    view_size: Size<f64, Logical>,

    /// Scale of the output the layer surface is on (and rounds its sizes to).
    scale: f64,

    /// Clock for driving animations.
    clock: Clock,
}

niri_render_elements! {
    LayerSurfaceRenderElement<R> => {
        Wayland = WaylandSurfaceRenderElement<R>,
        SolidColor = SolidColorRenderElement,
        Shadow = ShadowRenderElement,
        BackgroundEffect = BackgroundEffectElement,
    }
}

impl MappedLayer {
    pub fn new(
        surface: LayerSurface,
        rules: ResolvedLayerRules,
        view_size: Size<f64, Logical>,
        scale: f64,
        clock: Clock,
        config: &Config,
    ) -> Self {
        let mut shadow_config = config.layout.shadow;
        // Shadows for layer surfaces need to be explicitly enabled.
        shadow_config.on = false;
        shadow_config.merge_with(&rules.shadow);

        Self {
            surface,
            rules,
            block_out_buffer: SolidColorBuffer::new((0., 0.), [0., 0., 0., 1.]),
            view_size,
            scale,
            shadow: Shadow::new(shadow_config),
            blur: BlurElement::new(),
            clock,
        }
    }

    pub fn update_config(&mut self, config: &Config) {
        let mut shadow_config = config.layout.shadow;
        // Shadows for layer surfaces need to be explicitly enabled.
        shadow_config.on = false;
        shadow_config.merge_with(&self.rules.shadow);
        self.shadow.update_config(shadow_config);

        self.blur.update_config(config.blur);
    }

    pub fn update_shaders(&mut self) {
        self.shadow.update_shaders();
    }

    pub fn update_sizes(&mut self, view_size: Size<f64, Logical>, scale: f64) {
        self.view_size = view_size;
        self.scale = scale;
    }

    pub fn update_render_elements(&mut self, size: Size<f64, Logical>) {
        // Round to physical pixels.
        let size = size
            .to_physical_precise_round(self.scale)
            .to_logical(self.scale);

        self.block_out_buffer.resize(size);

        let radius = self.rules.geometry_corner_radius.unwrap_or_default();
        // FIXME: is_active based on keyboard focus?
        self.shadow
            .update_render_elements(size, true, radius, self.scale, 1.);

        self.blur.update_render_elements(self.scale, radius);
    }

    pub fn are_animations_ongoing(&self) -> bool {
        self.rules.baba_is_float
    }

    pub fn surface(&self) -> &LayerSurface {
        &self.surface
    }

    pub fn rules(&self) -> &ResolvedLayerRules {
        &self.rules
    }

    /// Recomputes the resolved layer rules and returns whether they changed.
    pub fn recompute_layer_rules(&mut self, rules: &[LayerRule], is_at_startup: bool) -> bool {
        let new_rules = ResolvedLayerRules::compute(rules, &self.surface, is_at_startup);
        if new_rules == self.rules {
            return false;
        }

        self.rules = new_rules;
        true
    }

    pub fn place_within_backdrop(&self) -> bool {
        if !self.rules.place_within_backdrop {
            return false;
        }

        if self.surface.layer() != Layer::Background {
            return false;
        }

        let state = self.surface.cached_state();
        if state.exclusive_zone != ExclusiveZone::DontCare {
            return false;
        }

        true
    }

    pub fn bob_offset(&self) -> Point<f64, Logical> {
        if !self.rules.baba_is_float {
            return Point::from((0., 0.));
        }

        let y = baba_is_float_offset(self.clock.now(), self.view_size.h);
        let y = round_logical_in_physical(self.scale, y);
        Point::from((0., y))
    }

    pub fn render_normal<R: NiriRenderer>(
        &self,
        mut ctx: RenderCtx<R>,
        location: Point<f64, Logical>,
        mut pos_in_backdrop: Point<f64, Logical>,
        zoom: f64,
        push: &mut dyn FnMut(LayerSurfaceRenderElement<R>),
    ) {
        let scale = Scale::from(self.scale);
        let alpha = self.rules.opacity.unwrap_or(1.).clamp(0., 1.);
        let location = location + self.bob_offset();
        pos_in_backdrop += self.bob_offset().upscale(zoom);

        if ctx.target.should_block_out(self.rules.block_out_from) {
            // Round to physical pixels.
            let location = location.to_physical_precise_round(scale).to_logical(scale);

            // FIXME: take geometry-corner-radius into account.
            let elem = SolidColorRenderElement::from_buffer(
                &self.block_out_buffer,
                location,
                alpha,
                Kind::Unspecified,
            );
            push(elem.into());
        } else {
            // Layer surfaces don't have extra geometry like windows.
            let buf_pos = location;

            let surface = self.surface.wl_surface();
            push_elements_from_surface_tree(
                ctx.renderer,
                surface,
                buf_pos.to_physical_precise_round(scale),
                scale,
                alpha,
                Kind::ScanoutCandidate,
                &mut |elem| push(elem.into()),
            );
        }

        let location = location.to_physical_precise_round(scale).to_logical(scale);
        self.shadow
            .render(ctx.renderer, location, &mut |elem| push(elem.into()));

        let effect = self.rules.background_effect;
        let area = Rectangle::new(location, self.block_out_buffer.size());
        let mut blur = effect.blur == Some(true);
        // Effects not requested by the surface itself are drawn to match the geometry.
        let mut clip_to_geometry = true;

        // FIXME: support blur regions on subsurfaces in addition to the main surface.
        let blur_geometry = if let Some(region) = self.blur_region() {
            let main_surface_geo = self.main_surface_geo();
            let region = match region {
                KdeBlurRegion::WholeSurface => Some(main_surface_geo),
                KdeBlurRegion::Region(region) => {
                    // FIXME: support regions with more than one rect.
                    let rect = region.rects.iter().copied().find_map(|(kind, rect)| {
                        matches!(kind, RectangleKind::Add).then_some(rect)
                    });
                    rect.and_then(|rect| {
                        rect.intersection(Rectangle::from_size(main_surface_geo.size))
                    })
                    .map(|mut rect| {
                        rect.loc += main_surface_geo.loc;
                        rect
                    })
                }
            };

            if let Some(region) = region {
                // If the surface itself requests the effects, apply different defaults.
                blur = effect.blur != Some(false);
                clip_to_geometry = false;

                trace!("rendering layer ext/kde blur with region={region:?}");
                let mut region = region.to_f64();
                region.loc += area.loc;
                Some(region)
            } else {
                None
            }
        } else {
            Some(area)
        };

        if let Some(geometry) = blur_geometry {
            pos_in_backdrop += (geometry.loc - area.loc).upscale(zoom);
            // TODO: damage on parameter change
            let mut params = background_effect::Parameters {
                geometry,
                window_geometry: area,
                pos_in_backdrop,
                zoom,
                corner_radius: self.rules.geometry_corner_radius.unwrap_or_default(),
                scale: self.scale as f32,
                xray: effect.xray == Some(true),
                blur,
                noise: effect.noise,
                saturation: effect.saturation,
                clip_to_geometry,
            };
            // If we have some background effect but xray wasn't explicitly set, default it to true
            // since it's cheaper.
            if params.is_visible() && effect.xray.is_none() {
                params.xray = true;
            }
            background_effect::render(ctx.as_gles(), params, &self.blur, &mut |elem| {
                push(elem.into())
            });
        }
    }

    pub fn render_popups<R: NiriRenderer>(
        &self,
        ctx: RenderCtx<R>,
        location: Point<f64, Logical>,
        push: &mut dyn FnMut(LayerSurfaceRenderElement<R>),
    ) {
        let scale = Scale::from(self.scale);
        let alpha = self.rules.opacity.unwrap_or(1.).clamp(0., 1.);
        let location = location + self.bob_offset();

        if ctx.target.should_block_out(self.rules.block_out_from) {
            return;
        }

        // Layer surfaces don't have extra geometry like windows.
        let buf_pos = location;

        let surface = self.surface.wl_surface();
        for (popup, popup_offset) in PopupManager::popups_for_surface(surface) {
            // Layer surfaces don't have extra geometry like windows.
            let offset = popup_offset - popup.geometry().loc;

            push_elements_from_surface_tree(
                ctx.renderer,
                popup.wl_surface(),
                (buf_pos + offset.to_f64()).to_physical_precise_round(scale),
                scale,
                alpha,
                Kind::ScanoutCandidate,
                &mut |elem| push(elem.into()),
            );
        }
    }

    fn main_surface_geo(&self) -> Rectangle<i32, Logical> {
        with_states(self.surface.wl_surface(), |states| {
            let data = states.data_map.get::<RendererSurfaceStateUserData>();
            data.and_then(|d| d.lock().unwrap().view())
                .map(|view| Rectangle {
                    loc: view.offset,
                    size: view.dst,
                })
        })
        .unwrap_or_default()
    }

    fn blur_region(&self) -> Option<KdeBlurRegion> {
        with_states(self.surface.wl_surface(), |states| {
            let cached = &states.cached_state;

            // Prefer ext-background-effect.
            if cached.has::<BackgroundEffectSurfaceCachedState>() {
                let mut guard = cached.get::<BackgroundEffectSurfaceCachedState>();
                guard
                    .current()
                    .blur_region
                    .clone()
                    .map(KdeBlurRegion::Region)
            } else {
                let mut guard = cached.get::<KdeBlurSurfaceCachedState>();
                guard.current().blur_region.clone()
            }
        })
    }
}
