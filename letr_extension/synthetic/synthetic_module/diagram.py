from numpy.random import uniform, choice
from random import randint, choice as rand_choice
import numpy as np
from helper.seed import use_seed
from helper.text import get_dictionary, TEXT_COLORED_FREQ
from element import AbstractElement
from PIL import Image, ImageDraw, ImageFilter
import svgwrite

POS_ELEMENT_OPACITY_RANGE = {
    "drawing": (220, 255),
    "glyph": (150, 255),
    "image": (150, 255),
    "table": (200, 255),
    "line": (120, 200),
    "table_word": (50, 200),
    "text": (200, 255),
    "diagram": (180, 255),
}

NEG_ELEMENT_OPACITY_RANGE = {
    "drawing": (0, 10),
    "glyph": (0, 10),
    "image": (0, 25),
    "table": (0, 25),
    "text": (0, 10),
    "diagram": (0, 25),
}
NEG_ELEMENT_BLUR_RADIUS_RANGE = (1, 2.5)
WIDTH_VALUES = [1, 2, 3]

DIAGRAM_COLOR = (255, 100, 180)

COCENTRIC_CIRCLES_RATIO = 0.1
SAME_RADIUS_CIRCLES_RATIO = 0.1


class DiagramElement(AbstractElement):
    color = DIAGRAM_COLOR
    name = "diagram"

    @use_seed()
    def generate_content(self):
        dictionary, self.font = get_dictionary(self.parameters, self.height)
        self.diagram_position = self.parameters["diagram_position"]
        self.as_negative = self.parameters.get("as_negative", False)
        self.blur_radius = (
            uniform(*NEG_ELEMENT_BLUR_RADIUS_RANGE) if self.as_negative else None
        )
        self.opacity = randint(
            *NEG_ELEMENT_OPACITY_RANGE[self.name]
            if self.as_negative
            else POS_ELEMENT_OPACITY_RANGE[self.name]
        )
        self.colored = choice(
            [True, False], p=[TEXT_COLORED_FREQ, 1 - TEXT_COLORED_FREQ]
        )
        self.colors = (
            tuple([randint(0, 60)] * 3)
            if not self.colored
            else tuple([randint(0, 255) for _ in range(3)])
        )
        self.colors_alpha = self.colors + (self.opacity,)
        self.number_circles = int(max((np.random.normal(randint(4, 15), 2)), 1))
        self.number_arcs = 0
        self.number_lines = int(max((np.random.normal(randint(4, 30), 2)), 1))
        self.number_words = int(max((np.random.normal(randint(4, 30), 2)), 0))

        self.table, self.content_width, self.content_height = self._generate_diagram(
            dictionary
        )
        self.pos_x = randint(self.diagram_position[0], self.width - self.content_width)
        self.pos_y = randint(
            self.diagram_position[1], self.height - self.content_height
        )

    def _generate_diagram(self, dictionary):
        width = randint(
            max(self.diagram_position[0], self.width // 4),
            self.width - 2 * self.diagram_position[0],
        )
        height = randint(
            max(self.diagram_position[1], self.height // 4),
            self.height - 2 * self.diagram_position[1],
        )
        # FIXME: diagram can be smaller than the background but with no less than 200 pixels

        circle_pos, circle_radius = [], []
        for i in range(self.number_circles):
            radius = randint(
                min(10, min(width, height) // 2.1), min(width, height) // 2.1
            )
            center_x = np.random.randint(radius, width - radius)
            center_y = np.random.randint(radius, height - radius)
            circle_radius.append(radius)
            circle_pos.append((center_x, center_y))
            add_cocentric_circles = choice(
                [True, False], p=[COCENTRIC_CIRCLES_RATIO, 1 - COCENTRIC_CIRCLES_RATIO]
            )
            if add_cocentric_circles and radius > (min(width, height) // 6):
                num_circles_2 = 2 * randint(2, 8)

                for k in range(0, num_circles_2, 3):
                    new_radius = radius * np.random.uniform(
                        (k + 1) / num_circles_2, min((k + 2) / num_circles_2, 1)
                    )
                    circle_radius.append(new_radius)
                    circle_pos.append((center_x, center_y))

            add_same_radius_circles = choice(
                [True, False],
                p=[SAME_RADIUS_CIRCLES_RATIO, 1 - SAME_RADIUS_CIRCLES_RATIO],
            )
            if add_same_radius_circles:
                num_circles_2 = 2 * randint(2, 10)

                for k in range(0, num_circles_2, 2):
                    skip = choice([True, False], p=[0.5, 0.5])
                    if skip:
                        continue
                    new_angle = np.random.uniform(
                        2 * np.pi * (k + 1) / num_circles_2,
                        2 * (k + 2) * np.pi / num_circles_2,
                    )
                    new_center_x = center_x + radius * np.cos(new_angle)
                    new_center_y = center_y + radius * np.sin(new_angle)

                    circle_radius.append(radius)
                    circle_pos.append((new_center_x, new_center_y))


        arc_pos, arc_radius, arc_angles = [], [], []
        for i in range(self.number_arcs):
            radius = randint(min(5, min(width, height) // 2), min(width, height) // 2)
            arc_radius.append(radius)
            center_x = np.random.randint(radius, width - radius)
            center_y = np.random.randint(radius, height - radius)
            arc_pos.append((center_x, center_y))
            arc_angle_1 = np.random.uniform(0, 360)
            arc_angle_2 = np.random.uniform(0, 360)
            arc_angles.append((arc_angle_1, arc_angle_2))


        line_coords = []
        for i in range(self.number_lines):
            length = randint(10, min(width // 2 - 1, height // 2 - 1))
            coords_x = np.random.randint(length, width - length)
            coords_y = np.random.randint(length, height - length)

            angle = np.random.uniform(0, 2 * np.pi)
            x_length = length * np.abs(np.cos(angle))
            y_length = length * np.sin(angle)
            direction = choice([1, -1])
            coords = (
                coords_x - x_length,
                coords_y - direction * y_length,
                coords_x + x_length,
                coords_y + direction * y_length,
            )
            line_coords.append(coords)
        words, word_positions = [], []
        for i in range(self.number_words):
            word_as_number = choice([True, False])

            if word_as_number:
                n_letter = randint(1, 4)
                word = f"{randint(0, 10**n_letter - 1):,}"

            else:
                word = rand_choice(dictionary)

                uppercase = choice([True, False])
                if uppercase:
                    word = word.upper()

            if len(word) > 0:
                w, h = self.font.getsize(word)
                try:
                    top_left_x = np.random.randint(0, width - w)
                    top_left_y = np.random.randint(0, height - h)
                except ValueError: # word is too long for the diagram
                    continue
                words.append(word)
                word_positions.append((top_left_x, top_left_y))

        return (
            {
                "circle_pos": circle_pos,
                "circle_radius": circle_radius,
                "line_coords": line_coords,
                "arc_pos": arc_pos,
                "arc_radius": arc_radius,
                "arc_angles": arc_angles,
                "words": words,
                "word_positions": word_positions,
            },
            width,
            height,
        )

    def to_image(self):
        canvas = Image.new("RGBA", self.size)
        draw = ImageDraw.Draw(canvas)
        fill = choice([False, True], p=[0.8, 0.2])
        if fill:
            opacity = randint(40, 80)
            fill_color = tuple(randint(0, 255) for _ in range(3)) + (opacity,)
        else:
            fill_color = None
        prev_circle_radius = 0
        prev_circle_pos = (0, 0, 0, 0)
        for circle_pos, circle_radius in zip(
            self.table["circle_pos"], self.table["circle_radius"]
        ):
            keep_same_params = (prev_circle_radius == circle_radius) or (
                prev_circle_pos == circle_pos
            )
            if not keep_same_params:
                params = {
                    "fill": fill_color,
                    "outline": self.colors,
                    "width": rand_choice(WIDTH_VALUES),
                }

            center = [self.pos_x + circle_pos[0], self.pos_y + circle_pos[1]]
            shape = [
                center[0] - circle_radius,
                center[1] - circle_radius,
                center[0] + circle_radius,
                center[1] + circle_radius,
            ]
            fill_color = None  # only fill the first circle to not overlap
            draw.ellipse(shape, **params)
            prev_circle_radius = circle_radius
            prev_circle_pos = circle_pos
        for arc_pos, arc_radius, arc_angles in zip(
            self.table["arc_pos"], self.table["arc_radius"], self.table["arc_angles"]
        ):

            params = {
                "fill": self.colors,
                "width": rand_choice(WIDTH_VALUES),
            }

            center = [self.pos_x + arc_pos[0], self.pos_y + arc_pos[1]]
            shape = [
                center[0] - arc_radius,
                center[1] - arc_radius,
                center[0] + arc_radius,
                center[1] + arc_radius,
            ]

            draw.arc(shape, start=arc_angles[0], end=arc_angles[1], **params)

        for line_coords in self.table["line_coords"]:

            params = {
                "fill": self.colors,
                "width": rand_choice(WIDTH_VALUES),
            }

            draw.line(
                [
                    self.pos_x + line_coords[0],
                    self.pos_y + line_coords[1],
                    self.pos_x + line_coords[2],
                    self.pos_y + line_coords[3],
                ],
                **params,
            )
            assert self.pos_x + line_coords[0] < self.width, f"{line_coords}"
            assert self.pos_y + line_coords[1] < self.height, f"{line_coords}"
            assert self.pos_x + line_coords[2] < self.width, f"{line_coords}"
            assert self.pos_y + line_coords[3] < self.height, f"{line_coords}"

        for word, pos in zip(self.table["words"], self.table["word_positions"]):
            opacity = randint(*POS_ELEMENT_OPACITY_RANGE[self.name])

            colors_alpha = self.colors
            pos = pos[0] + self.pos_x, pos[1] + self.pos_y

            draw.text(pos, word, font=self.font, fill=colors_alpha)

        if self.as_negative:
            canvas = canvas.filter(ImageFilter.GaussianBlur(self.blur_radius))

        return canvas

    def to_svg(self, output_svg_path, background_path=None):
        drawing = svgwrite.Drawing(str(output_svg_path), size=self.size)
        if background_path is not None:
            drawing.add(
                drawing.image(href=str(background_path), insert=(0, 0), size=self.size)
            )
        opacity = 1
        self.offset = [
            self.pos_x,
            self.pos_y,
        ]
        for circle_pos, circle_radius in zip(
            self.table["circle_pos"], self.table["circle_radius"]
        ):
            params = {
                "fill": "none",
                "stroke": "black",
                "stroke_width": 1,
                "stroke_opacity": opacity,
            }  
            center = [self.offset[0] + circle_pos[0], self.offset[1] + circle_pos[1]]
            drawing.add(
                drawing.circle(
                    center=center,
                    r=circle_radius,
                    **params,
                )
            )
        for arc_pos, arc_radius, arc_angle in zip(
            self.table["arc_pos"], self.table["arc_radius"], self.table["arc_angles"]
        ):
            params = {
                "fill": "none",
                "stroke": "black",
                "stroke_width": 1,
                "stroke_opacity": opacity,
            }

            center = [
                self.offset[0] + arc_pos[0],
                self.offset[1] + arc_pos[1],
            ]
            rad_angle = np.array(arc_angle) * np.pi / 180
            p0 = (
                arc_radius * np.array([np.cos(rad_angle[0]), np.sin(rad_angle[0])])
                + center
            )
            p1 = (
                arc_radius * np.array([np.cos(rad_angle[1]), np.sin(rad_angle[1])])
                + center
            )

            def is_large_arc(rad_angle):
                if rad_angle[0] <= np.pi:
                    return not (rad_angle[0] < rad_angle[1] < (np.pi + rad_angle[0]))
                return (rad_angle[0] - np.pi) < rad_angle[1] < rad_angle[0]

            arc_args = {
                "x0": p0[0],
                "y0": p0[1],
                "xradius": arc_radius,
                "yradius": arc_radius,
                "ellipseRotation": 0,  # has no effect for circles
                "x1": (p1[0] - p0[0]),
                "y1": (p1[1] - p0[1]),
                "large_arc_flag": int(is_large_arc(rad_angle)),
                "sweep_flag": 1,  # set sweep-flag to 1 for clockwise arc
            }
            drawing.add(
                drawing.path(
                    d="M %(x0)f,%(y0)f a %(xradius)f,%(yradius)f %(ellipseRotation)f %(large_arc_flag)d,%(sweep_flag)d %(x1)f,%(y1)f"
                    % arc_args,
                    fill="none",
                    stroke="red",
                    stroke_width=1,
                )
            )

        for line_coords in self.table["line_coords"]:
            params = {"stroke": "black", "stroke_width": 1, "stroke-opacity": opacity}
            p1 = [self.offset[0] + line_coords[0], self.offset[1] + line_coords[1]]
            p2 = [self.offset[0] + line_coords[2], self.offset[1] + line_coords[3]]

            drawing.add(drawing.line(start=p1, end=p2, **params))
        for word, pos in zip(self.table["words"], self.table["word_positions"]):
            opacity = randint(
                *NEG_ELEMENT_OPACITY_RANGE[self.name]
                if self.as_negative
                else (100, 200)
            )


            pos = (pos[0] + self.offset[0], pos[1] + self.offset[1])
            drawing.add(drawing.text(word, insert=pos))

        drawing.save()

    def get_annotation(self):
        centers, circle_radii, lines = [], [], []
        arc_centers, arc_radii, arc_angles = [], [], []
        self.offset = [self.pos_x, self.pos_y]
        for circle_pos, circle_radius in zip(
            self.table["circle_pos"], self.table["circle_radius"]
        ):
            center = [self.offset[0] + circle_pos[0], self.offset[1] + circle_pos[1]]
            centers.append(center)
            circle_radii.append(circle_radius)
        for line_coords in self.table["line_coords"]:
            lines.append(
                [
                    int(self.offset[0] + line_coords[0]),
                    int(self.offset[1] + line_coords[1]),
                    int(self.offset[0] + line_coords[2]),
                    int(self.offset[1] + line_coords[3]),
                ]
            )
        for arc_pos, arc_radius, arc_angle in zip(
            self.table["arc_pos"], self.table["arc_radius"], self.table["arc_angles"]
        ):
            arc_centers.append(
                [self.offset[0] + arc_pos[0], self.offset[1] + arc_pos[1]]
            )
            arc_radii.append(arc_radius)
            arc_angles.append(arc_angle)

        return {
            "circle_centers": centers,
            "circle_radii": circle_radii,
            "lines": lines,
            "arc_centers": arc_centers,
            "arc_angles": arc_angles,
            "arc_radii": arc_radii,
        }
