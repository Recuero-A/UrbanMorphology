extends Node3D
## UrbanMorph Milestone 11 viewer: loads a city GLB at runtime and lets you fly or walk through it.
## Controls: mouse look | WASD move | Space / Shift up / down (fly) or jump (walk) | F toggle fly / walk |
## mouse wheel or +/- change speed | Esc release / capture mouse | Q quit.
## Units: metres, Y up (glTF). A city tile spans x 0..512, z -512..0 (city north = -z).

var args := {}
var city: Node3D
var cam: Camera3D
var body: CharacterBody3D
var walk_cam: Camera3D
var flying := true
var speed := 12.0
var yaw := 0.0
var pitch := 0.0
var bench_t := -1.0
var bench_len := 0.0
var frame_ms: Array[float] = []
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


func add_collision() -> void:
	var t0 := Time.get_ticks_msec()
	var tris := 0
	var meshes := mesh_instances(city, [])
	for mi in meshes:
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
	var sun := DirectionalLight3D.new()
	sun.rotation_degrees = Vector3(-42, -140, 0)
	sun.light_energy = 1.2
	sun.shadow_enabled = true
	sun.directional_shadow_max_distance = 600.0
	add_child(sun)
	var env := Environment.new()
	var sky := Sky.new()
	sky.sky_material = ProceduralSkyMaterial.new()
	env.background_mode = Environment.BG_SKY
	env.sky = sky
	env.ambient_light_source = Environment.AMBIENT_SOURCE_SKY
	env.tonemap_mode = Environment.TONE_MAPPER_AGX
	env.ssao_enabled = true
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
	add_collision()
	setup_environment()
	var sl := read_viewer_json()
	setup_cameras(sl[0], sl[1])
	if args.has("check"):
		await get_tree().physics_frame
		await get_tree().physics_frame
		run_check(sl[0])
		return
	if args.has("benchmark"):
		bench_len = float(args["benchmark"])
		bench_t = 0.0
	else:
		Input.mouse_mode = Input.MOUSE_MODE_CAPTURED


func ray_down(x: float, z: float) -> Dictionary:
	var q := PhysicsRayQueryParameters3D.create(Vector3(x, 500, z), Vector3(x, -10, z))
	return get_world_3d().direct_space_state.intersect_ray(q)


func run_check(spawn: Vector3) -> void:
	var hits := 0
	var n := 0
	var ground_hits := 0
	for i in range(8):
		for j in range(8):
			var r := ray_down(16 + i * 64, -(16 + j * 64))
			n += 1
			if not r.is_empty():
				hits += 1
				if r.position.y < 0.5:
					ground_hits += 1
	var aabb := AABB()
	var first := true
	for mi in mesh_instances(city, []):
		var b: AABB = mi.global_transform * mi.get_aabb()
		aabb = b if first else aabb.merge(b)
		first = false
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
			speed = min(speed * 1.25, 400.0)
		elif e.button_index == MOUSE_BUTTON_WHEEL_DOWN:
			speed = max(speed / 1.25, 1.0)
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
			KEY_EQUAL, KEY_KP_ADD:
				speed = min(speed * 1.25, 400.0)
			KEY_MINUS, KEY_KP_SUBTRACT:
				speed = max(speed / 1.25, 1.0)
			KEY_Q:
				get_tree().quit()


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
	var d := move_dir(yaw) * minf(speed, 8.0)
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
		info["draw_calls_last_frame"] = RenderingServer.get_rendering_info(RenderingServer.RENDERING_INFO_TOTAL_DRAW_CALLS_IN_FRAME)
		info["primitives_last_frame"] = RenderingServer.get_rendering_info(RenderingServer.RENDERING_INFO_TOTAL_PRIMITIVES_IN_FRAME)
		info["video_mem_mb"] = RenderingServer.get_rendering_info(RenderingServer.RENDERING_INFO_VIDEO_MEM_USED) / 1048576.0
		info["static_mem_mb"] = OS.get_static_memory_usage() / 1048576.0
		info["window"] = [get_viewport().size.x, get_viewport().size.y]
		info["godot_version"] = Engine.get_version_info()["string"]
		info["renderer"] = RenderingServer.get_video_adapter_name()
		write_out()
		get_tree().quit(0)
