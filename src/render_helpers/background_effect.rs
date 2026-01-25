use niri_config::CornerRadius;
use smithay::backend::renderer::gles::GlesRenderer;
use smithay::utils::{Logical, Point, Rectangle};

use crate::niri_render_elements;
use crate::render_helpers::blur::BlurElement;
use crate::render_helpers::xray::XrayElement;
use crate::render_helpers::RenderCtx;

niri_render_elements! {
    BackgroundEffectElement => {
        Blur = BlurElement,
        Xray = XrayElement,
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct Parameters {
    pub geometry: Rectangle<f64, Logical>,
    pub window_geometry: Rectangle<f64, Logical>,
    pub pos_in_backdrop: Point<f64, Logical>,
    /// Zoom factor between backdrop coordinates and geometry.
    pub zoom: f64,
    pub corner_radius: CornerRadius,
    pub scale: f32,
    pub xray: bool,
    pub blur: bool,
    pub noise: Option<f64>,
    pub saturation: Option<f64>,
    pub clip_to_geometry: bool,
}

impl Parameters {
    pub fn is_visible(&self) -> bool {
        self.xray
            || self.blur
            || self.noise.is_some_and(|x| x > 0.)
            || self.saturation.is_some_and(|x| x != 1.)
    }

    fn fit_radius(&mut self) {
        self.corner_radius = self.corner_radius.fit_to(
            self.window_geometry.size.w as f32,
            self.window_geometry.size.h as f32,
        );
    }
}

pub fn render(
    ctx: RenderCtx<GlesRenderer>,
    mut params: Parameters,
    blur: &BlurElement,
    push: &mut dyn FnMut(BackgroundEffectElement),
) {
    if !params.is_visible() {
        return;
    }

    if !params.clip_to_geometry {
        // Matching window geometry to geometry effectively prevents any clipping.
        params.window_geometry = params.geometry;
        params.corner_radius = CornerRadius::default();
    }

    if params.xray {
        let Some(xray) = ctx.xray else {
            return;
        };

        params.fit_radius();

        xray.render(ctx, params, &mut |elem| push(elem.into()));
    } else {
        // Render the non-xray blur.
        if let Some(elem) = blur.render(ctx.renderer, params) {
            push(elem.into());
        }
    }
}
