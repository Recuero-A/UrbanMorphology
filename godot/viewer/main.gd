extends Node3D
## UrbanMorph Milestone 11 viewer: loads a city GLB at runtime and lets you fly or walk through it.
## Controls: mouse look | WASD move | Space / Shift up / down (fly) or jump / run (walk) | F toggle fly / walk |
## mouse wheel or +/- change the speed of the current mode (shown on screen) | Esc release / capture mouse | Q quit.
## Units: metres, Y up (glTF). A city tile spans x 0..512, z -512..0 (city north = -z).

var args := {}
var city: Node3D
var cam: Camera3D
var body: CharacterBody3D
var walk_cam: Camera3D
var flying := true
var speed := 12.0                       # fly (m/s)
var walk_speed := 8.0                   # walk (m/s); Shift runs at RUN x
const RUN := 3.0
var hud: Label
var hud_t := 0.0
var yaw := 0.0
var pitch := 0.0
var bench_t := -1.0
var bench_len := 0.0
var frame_ms: Array[float] = []
var gpu_ms: Array[float] = []
var cpu_ms: Array[float] = []
var info := {}


func parse_args() -> void:
	var a := OS.get_cmdline_user_args()
	var i := 0
	while i < a.size():
		var k: String = a[i].trim_prefix("--")
		if i + 1 < a.size() and not a[i + 1].begins_with("--"):
			args[k] = a[i + 1]
			i += 2
		else:
			args[k] = true
			i += 1


func load_city(path: String) -> Node3D:
	var doc := GLTFDocument.new()
	var state := GLTFState.new()
	var t0 := Time.get_ticks_msec()
	var err := doc.append_from_file(path, state)
	if err != OK:
		push_error("GLB load failed: %s (%d)" % [path, err])
		return null
	var scene: Node3D = doc.generate_scene(state)
	info["load_ms"] = Time.get_ticks_msec() - t0
	return scene


func mesh_instances(n: Node, out: Array) -> Array:
	if n is MeshInstance3D:
		out.append(n)
	for c in n.get_children():
		mesh_instances(c, out)
	return out


## Themes: plants and lamps arrive as thousands of nodes sharing a few meshes (glTF linked duplicates). Each shared
## mesh becomes MultiMeshInstance3D chunks of CHUNK_M x CHUNK_M (frustum culling and visibility ranges per chunk);
## small plants fade out beyond SMALL_END_M and cast no shadows.
const CHUNK_M := 128.0
const SMALL := ["flowers", "shrub", "bougainvillea", "banana", "lamp"]
const SMALL_END_M := 380.0


func optimize_instances() -> void:
	var t0 := Time.get_ticks_msec()
	var by_mesh := {}
	for mi in mesh_instances(city, []):
		if mi.mesh == null:
			continue
		if not by_mesh.has(mi.mesh):
			by_mesh[mi.mesh] = []
		by_mesh[mi.mesh].append(mi)
	var n_mm := 0
	var n_inst := 0
	for mesh in by_mesh:
		var lst: Array = by_mesh[mesh]
		if lst.size() < 8:
			continue
		var kind := String(lst[0].name).rstrip("0123456789").trim_suffix("_")
		var small := kind in SMALL
		var cells := {}
		for mi in lst:
			var p: Vector3 = mi.global_position
			var key := Vector2i(int(floor(p.x / CHUNK_M)), int(floor(p.z / CHUNK_M)))
			if not cells.has(key):
				cells[key] = []
			cells[key].append(mi.global_transform)
		for key in cells:
			var xs: Array = cells[key]
			var mm := MultiMesh.new()
			mm.transform_format = MultiMesh.TRANSFORM_3D
			mm.mesh = mesh
			mm.instance_count = xs.size()
			for k in xs.size():
				mm.set_instance_transform(k, xs[k])
			var mmi := MultiMeshInstance3D.new()
			mmi.multimesh = mm
			mmi.name = "%s_mm_%d_%d" % [kind, key.x, key.y]
			if small:
				mmi.visibility_range_end = SMALL_END_M
				mmi.visibility_range_end_margin = 40.0
				mmi.visibility_range_fade_mode = GeometryInstance3D.VISIBILITY_RANGE_FADE_SELF
				mmi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
			add_child(mmi)
			n_mm += 1
		n_inst += lst.size()
		for mi in lst:
			mi.get_parent().remove_child(mi)
			mi.queue_free()
	info["multimesh_chunks"] = n_mm
	info["instances_merged"] = n_inst
	info["instancing_ms"] = Time.get_ticks_msec() - t0


func add_collision() -> void:
	var t0 := Time.get_ticks_msec()
	var tris := 0
	var meshes := mesh_instances(city, [])
	for mi in meshes:
		if String(mi.name).begins_with("glass") or String(mi.name).begins_with("carpentry") or String(mi.name).begins_with("metal"):
			continue                              # thin ornament: no collision needed to walk the streets
		var m: Mesh = mi.mesh
		for s in m.get_surface_count():
			var arr := m.surface_get_arrays(s)
			var idx = arr[Mesh.ARRAY_INDEX]
			tris += (idx.size() if idx != null else arr[Mesh.ARRAY_VERTEX].size()) / 3
		mi.create_trimesh_collision()
	info["mesh_instances"] = meshes.size()
	info["triangles"] = tris
	info["collision_ms"] = Time.get_ticks_msec() - t0


func setup_environment() -> void:
	## --light sunset (golden hour: low warm sun, orange horizon, warm aerial haze) | midday | (none: Milestone 11)
	var light := String(args.get("light", ""))
	var sun := DirectionalLight3D.new()
	sun.rotation_degrees = Vector3(-42, -140, 0)
	sun.light_energy = 1.2
	sun.shadow_enabled = true
	sun.directional_shadow_max_distance = 600.0
	add_child(sun)
	var env := Environment.new()
	var sky := Sky.new()
	var psky := ProceduralSkyMaterial.new()
	sky.sky_material = psky
	env.background_mode = Environment.BG_SKY
	env.sky = sky
	env.ambient_light_source = Environment.AMBIENT_SOURCE_SKY
	env.tonemap_mode = Environment.TONE_MAPPER_AGX
	env.ssao_enabled = true
	if light == "sunset":
		sun.rotation_degrees = Vector3(-8, -140, 0)          # themes/latin_caribbean/RESEARCH.md §6.2
		sun.light_color = Color("#ffb46b")
		sun.light_energy = 1.7
		psky.sky_top_color = Color("#6a5f93")              # dusky violet zenith over an amber horizon (IMG_3376)
		psky.sky_horizon_color = Color("#f3b95d")
		psky.sky_curve = 0.15
		psky.ground_horizon_color = Color("#c77a61")
		psky.ground_bottom_color = Color(0.25, 0.2, 0.18)
		psky.sun_angle_max = 20.0
		psky.sky_energy_multiplier = 1.1
		env.ambient_light_energy = 0.8
		env.fog_enabled = true
		env.fog_light_color = Color("#e9a36a")
		env.fog_density = 0.00018
		env.fog_aerial_perspective = 0.6
		env.fog_sky_affect = 0.25
		env.fog_sun_scatter = 0.25
		env.glow_enabled = true
		env.glow_intensity = 0.5
		env.adjustment_enabled = true
		env.adjustment_saturation = 1.12
	elif light == "midday":
		sun.rotation_degrees = Vector3(-62, -140, 0)
		sun.light_energy = 1.35
		env.fog_enabled = true
		env.fog_light_color = Color(0.75, 0.85, 1.0)
		env.fog_density = 0.00025
		env.fog_aerial_perspective = 0.5
		env.adjustment_enabled = true
		env.adjustment_saturation = 1.08
	var we := WorldEnvironment.new()
	we.environment = env
	add_child(we)


func setup_cameras(spawn: Vector3, look: Vector3) -> void:
	cam = Camera3D.new()
	cam.far = 4000.0
	cam.near = 0.1
	add_child(cam)
	cam.position = spawn + Vector3(0, 40, 0)
	cam.look_at(look, Vector3.UP)
	yaw = cam.rotation.y
	pitch = cam.rotation.x
	body = CharacterBody3D.new()
	var shape := CollisionShape3D.new()
	var capsule := CapsuleShape3D.new()
	capsule.radius = 0.35
	capsule.height = 1.8
	shape.shape = capsule
	body.add_child(shape)
	walk_cam = Camera3D.new()
	walk_cam.position = Vector3(0, 0.75, 0)
	walk_cam.far = 4000.0
	body.add_child(walk_cam)
	add_child(body)
	body.position = spawn + Vector3(0, 1.0, 0)
	cam.make_current()
	var layer := CanvasLayer.new()             # speed readout, shown for two seconds after a change
	hud = Label.new()
	hud.position = Vector2(16, 12)
	hud.add_theme_font_size_override("font_size", 22)
	hud.add_theme_color_override("font_outline_color", Color.BLACK)
	hud.add_theme_constant_override("outline_size", 6)
	hud.modulate.a = 0.0
	layer.add_child(hud)
	add_child(layer)


func add_sea() -> void:
	## Themes: viewer.json may carry sea_z; the sea then continues to the horizon (60 km plane just under the map's water).
	if not (args.has("viewer") and FileAccess.file_exists(args["viewer"])):
		return
	var d = JSON.parse_string(FileAccess.get_file_as_string(args["viewer"]))
	if d == null or not d.has("sea_z"):
		return
	var m := StandardMaterial3D.new()
	m.albedo_color = Color("#2a8c9a")
	m.roughness = 0.06
	m.metallic = 0.15
	var plane := PlaneMesh.new()
	plane.size = Vector2(60000, 60000)
	var mi := MeshInstance3D.new()
	mi.mesh = plane
	mi.material_override = m
	mi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	mi.position = Vector3(768, float(d["sea_z"]) - 0.06, -768)
	add_child(mi)


func read_viewer_json() -> Array:
	var spawn := Vector3(256, 1.7, -256)
	var look := Vector3(256, 1.7, -200)
	if args.has("viewer") and FileAccess.file_exists(args["viewer"]):
		var d = JSON.parse_string(FileAccess.get_file_as_string(args["viewer"]))
		if d != null:
			spawn = Vector3(d["spawn"][0], d["spawn"][1], d["spawn"][2])
			look = Vector3(d["look_at"][0], d["look_at"][1], d["look_at"][2])
	return [spawn, look]


func _ready() -> void:
	parse_args()
	if not args.has("glb"):
		push_error("usage: -- --glb /abs/path/scene.glb")
		get_tree().quit(2)
		return
	city = load_city(args["glb"])
	if city == null:
		get_tree().quit(1)
		return
	add_child(city)
	optimize_instances()
	add_collision()
	add_sea()
	setup_environment()
	var sl := read_viewer_json()
	setup_cameras(sl[0], sl[1])
	if args.has("shot"):                      # M16: screenshots (street level, oblique, aerial), then quit
		await take_shots(sl[0], sl[1], String(args["shot"]))
		get_tree().quit(0)
		return
	if args.has("check"):
		await get_tree().physics_frame
		await get_tree().physics_frame
		run_check(sl[0])
		return
	if args.has("benchmark"):
		bench_len = float(args["benchmark"])
		bench_t = 0.0
		RenderingServer.viewport_set_measure_render_time(get_viewport().get_viewport_rid(), true)
	else:
		Input.mouse_mode = Input.MOUSE_MODE_CAPTURED


func take_shots(spawn: Vector3, look: Vector3, prefix: String) -> void:
	var aabb := AABB()
	var first := true
	for mi in mesh_instances(city, []):
		var b: AABB = mi.global_transform * mi.get_aabb()
		aabb = b if first else aabb.merge(b)
		first = false
	var span := float(args.get("extent", 512.0))   # the city (x 0..extent, z -extent..0); the ground plane extends beyond
	var gr := ray_down(span / 2.0, -span / 2.0, aabb.end.y + 50.0, aabb.position.y - 10.0)
	var c := Vector3(span / 2.0, gr.position.y if not gr.is_empty() else spawn.y - 1.7, -span / 2.0)
	set_process(false)                       # the fly / walk controls would overwrite the camera every frame
	set_physics_process(false)
	cam.far = 6000.0
	cam.make_current()
	var views := {
		"street": [spawn + Vector3(0, 0.0, 0), look],
		"oblique": [c + Vector3(-0.62 * span, 0.42 * span, 0.62 * span), c],
		"aerial": [c + Vector3(0.0, 1.0 * span, 0.01), c],
	}
	for name in views:
		cam.global_position = views[name][0]
		cam.look_at(views[name][1], Vector3.UP)
		for k in range(int(args.get("frames", 12))):
			await get_tree().process_frame
		var img := get_viewport().get_texture().get_image()
		img.save_png(prefix + "_" + name + ".png")
		print("SHOT ", prefix + "_" + name + ".png")
	# themes: extra named views from viewer.json {"shots": {name: {"pos", "look", "fov" (vertical degrees)}}}
	var extra = {}
	if args.has("viewer") and FileAccess.file_exists(args["viewer"]):
		var d = JSON.parse_string(FileAccess.get_file_as_string(args["viewer"]))
		if d != null and d.has("shots"):
			extra = d["shots"]
	var fov0 := cam.fov
	for name in extra:
		var s = extra[name]
		cam.fov = float(s.get("fov", fov0))
		cam.global_position = Vector3(s["pos"][0], s["pos"][1], s["pos"][2])
		cam.look_at(Vector3(s["look"][0], s["look"][1], s["look"][2]), Vector3.UP)
		for k in range(int(args.get("frames", 12))):
			await get_tree().process_frame
		var img := get_viewport().get_texture().get_image()
		img.save_png(prefix + "_" + name + ".png")
		print("SHOT ", prefix + "_" + name + ".png")
	cam.fov = fov0


func ray_down(x: float, z: float, top: float, bottom: float) -> Dictionary:
	var q := PhysicsRayQueryParameters3D.create(Vector3(x, top, z), Vector3(x, bottom, z))
	return get_world_3d().direct_space_state.intersect_ray(q)


func run_check(spawn: Vector3) -> void:
	var aabb := AABB()
	var first := true
	for mi in mesh_instances(city, []):
		var b: AABB = mi.global_transform * mi.get_aabb()
		aabb = b if first else aabb.merge(b)
		first = false
	# 8 x 8 downward rays over the scene's footprint, cast from above its highest point (terrain may be far above 0)
	var hits := 0
	var n := 0
	var ground_hits := 0
	for i in range(8):
		for j in range(8):
			var x := aabb.position.x + (i + 0.5) * aabb.size.x / 8.0
			var z := aabb.position.z + (j + 0.5) * aabb.size.z / 8.0
			var r := ray_down(x, z, aabb.end.y + 50.0, aabb.position.y - 10.0)
			n += 1
			if not r.is_empty():
				hits += 1
				if r.position.y < 0.5:
					ground_hits += 1
	# drop the walking body from 1 m above the spawn and let it settle on the collision mesh
	body.position = spawn + Vector3(0, 1.0, 0)
	for k in range(120):
		body.velocity.y -= 9.8 / 60.0
		body.move_and_slide()
		await get_tree().physics_frame
	info["ray_hits"] = "%d/%d" % [hits, n]
	info["ray_hits_at_ground_level"] = ground_hits
	info["aabb_min"] = [aabb.position.x, aabb.position.y, aabb.position.z]
	info["aabb_max"] = [aabb.end.x, aabb.end.y, aabb.end.z]
	info["walker_on_floor"] = body.is_on_floor()
	info["walker_y"] = body.position.y
	info["godot_version"] = Engine.get_version_info()["string"]
	write_out()
	get_tree().quit(0)


func write_out() -> void:
	var s := JSON.stringify(info, " ")
	print("RESULT ", JSON.stringify(info))
	if args.has("out"):
		var f := FileAccess.open(args["out"], FileAccess.WRITE)
		f.store_string(s)
		f.close()


func _unhandled_input(e: InputEvent) -> void:
	if e is InputEventMouseMotion and Input.mouse_mode == Input.MOUSE_MODE_CAPTURED:
		yaw -= e.relative.x * 0.003
		pitch = clamp(pitch - e.relative.y * 0.003, -1.5, 1.5)
	elif e is InputEventMouseButton and e.pressed:
		if e.button_index == MOUSE_BUTTON_WHEEL_UP:
			change_speed(1.25)
		elif e.button_index == MOUSE_BUTTON_WHEEL_DOWN:
			change_speed(1.0 / 1.25)
		elif Input.mouse_mode != Input.MOUSE_MODE_CAPTURED:
			Input.mouse_mode = Input.MOUSE_MODE_CAPTURED
	elif e is InputEventKey and e.pressed and not e.echo:
		match e.keycode:
			KEY_ESCAPE:
				Input.mouse_mode = Input.MOUSE_MODE_VISIBLE
			KEY_F:
				flying = not flying
				if flying:
					cam.global_position = walk_cam.global_position
					cam.make_current()
				else:
					body.global_position = cam.global_position - Vector3(0, 0.75, 0)
					body.velocity = Vector3.ZERO
					walk_cam.make_current()
				show_speed()
			KEY_EQUAL, KEY_KP_ADD:
				change_speed(1.25)
			KEY_MINUS, KEY_KP_SUBTRACT:
				change_speed(1.0 / 1.25)
			KEY_Q:
				get_tree().quit()


func change_speed(k: float) -> void:
	if flying:
		speed = clampf(speed * k, 1.0, 400.0)
	else:
		walk_speed = clampf(walk_speed * k, 1.0, 30.0)
	show_speed()


func show_speed() -> void:
	if flying:
		hud.text = "fly %.0f m/s" % speed
	else:
		hud.text = "walk %.1f m/s  (Shift: run %.0f m/s)" % [walk_speed, walk_speed * RUN]
	hud_t = 2.0


func move_dir(basis_yaw: float) -> Vector3:
	var f := Vector3(-sin(basis_yaw), 0, -cos(basis_yaw))
	var r := Vector3(cos(basis_yaw), 0, -sin(basis_yaw))
	var d := Vector3.ZERO
	if Input.is_key_pressed(KEY_W): d += f
	if Input.is_key_pressed(KEY_S): d -= f
	if Input.is_key_pressed(KEY_D): d += r
	if Input.is_key_pressed(KEY_A): d -= r
	return d.normalized()


func _process(delta: float) -> void:
	if bench_t >= 0.0:
		benchmark_step(delta)
		return
	if hud_t > 0.0:
		hud_t -= delta
		hud.modulate.a = clampf(hud_t, 0.0, 1.0)
	if flying:
		cam.rotation = Vector3(pitch, yaw, 0)
		var d := move_dir(yaw)
		if Input.is_key_pressed(KEY_SPACE): d += Vector3.UP
		if Input.is_key_pressed(KEY_SHIFT): d -= Vector3.UP
		cam.position += d * speed * delta
	else:
		body.rotation.y = yaw
		walk_cam.rotation.x = pitch


func _physics_process(delta: float) -> void:
	if flying or bench_t >= 0.0 or args.has("check"):
		return
	var d := move_dir(yaw) * walk_speed * (RUN if Input.is_key_pressed(KEY_SHIFT) else 1.0)
	body.velocity.x = d.x
	body.velocity.z = d.z
	if body.is_on_floor():
		body.velocity.y = 4.5 if Input.is_key_pressed(KEY_SPACE) else 0.0
	else:
		body.velocity.y -= 9.8 * delta
	body.move_and_slide()


func benchmark_step(delta: float) -> void:
	## Fixed, deterministic camera path: an aerial orbit (first half), then a street-level glide (second half).
	bench_t += delta
	if bench_t > 1.0:                                # skip warm-up frames (shader compilation)
		frame_ms.append(delta * 1000.0)
		var rid := get_viewport().get_viewport_rid()
		gpu_ms.append(RenderingServer.viewport_get_measured_render_time_gpu(rid))
		cpu_ms.append(RenderingServer.viewport_get_measured_render_time_cpu(rid))
	var u: float = clamp(bench_t / bench_len, 0.0, 1.0)
	if u < 0.5:
		var a := u * 2.0 * TAU
		cam.position = Vector3(256 + 380 * cos(a), 220, -256 + 380 * sin(a))
		cam.look_at(Vector3(256, 0, -256), Vector3.UP)
	else:
		var sl := read_viewer_json()
		var p0: Vector3 = sl[0]
		var p1: Vector3 = sl[1]
		var dir := (p1 - p0).normalized()
		cam.position = p0 + dir * (u - 0.5) * 2.0 * 120.0
		cam.look_at(cam.position + dir, Vector3.UP)
	if bench_t >= bench_len:
		frame_ms.sort()
		var n := frame_ms.size()
		var mean := 0.0
		for x in frame_ms:
			mean += x
		mean /= max(n, 1)
		info["frames"] = n
		info["mean_frame_ms"] = mean
		info["mean_fps"] = 1000.0 / mean
		info["p95_frame_ms"] = frame_ms[int(0.95 * (n - 1))]
		info["fps_at_p95_frame"] = 1000.0 / frame_ms[int(0.95 * (n - 1))]
		gpu_ms.sort()
		cpu_ms.sort()
		info["gpu_render_ms_median"] = gpu_ms[gpu_ms.size() / 2]
		info["gpu_render_ms_p95"] = gpu_ms[int(0.95 * (gpu_ms.size() - 1))]
		info["cpu_render_ms_median"] = cpu_ms[cpu_ms.size() / 2]
		info["fps_capped_by_display"] = info["mean_fps"] > 118.0
		info["draw_calls_last_frame"] = RenderingServer.get_rendering_info(RenderingServer.RENDERING_INFO_TOTAL_DRAW_CALLS_IN_FRAME)
		info["primitives_last_frame"] = RenderingServer.get_rendering_info(RenderingServer.RENDERING_INFO_TOTAL_PRIMITIVES_IN_FRAME)
		info["video_mem_mb"] = RenderingServer.get_rendering_info(RenderingServer.RENDERING_INFO_VIDEO_MEM_USED) / 1048576.0
		info["static_mem_mb"] = OS.get_static_memory_usage() / 1048576.0
		info["window"] = [get_viewport().size.x, get_viewport().size.y]
		info["godot_version"] = Engine.get_version_info()["string"]
		info["renderer"] = RenderingServer.get_video_adapter_name()
		write_out()
		get_tree().quit(0)
