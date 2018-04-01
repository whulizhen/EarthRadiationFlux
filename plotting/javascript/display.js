
var scene, camera, renderer, origin, myearth;
var camRad = 20000;

window.onload = function()
{

	scene = new THREE.Scene();
	
	renderer = new THREE.WebGLRenderer();
	renderer.setClearColor(0xffffff,0);
	renderer.setSize( window.innerWidth, window.innerHeight );
	document.body.appendChild( renderer.domElement );
	
	origin = new THREE.Vector3( 0, 0, 0 );
	
	//******************** Camera and event handlers for moving camera
	camera = new THREE.PerspectiveCamera( 90, window.innerWidth / window.innerHeight, 0.1, 100000 );
	camera.up.set( 0, 0, 1 );
	camUpdate();
	
	document.addEventListener( 'mousemove', onDocumentMouseMove, false );
	document.addEventListener( 'mousedown', onDocumentMouseDown, false );
	document.addEventListener( 'mouseup', onDocumentMouseUp, false );
	document.addEventListener( 'mousewheel', onDocumentMouseWheel, false );
	document.addEventListener( 'DOMMouseScroll', onDocumentMouseWheel, false );
	
	//******************** Lighting
	var ambientLight = new THREE.AmbientLight(0x444444);
	scene.add(ambientLight);

	var sunlight = new THREE.DirectionalLight( 0xffffff, 1 );
	sunlight.position.set( 0, 1, 0.2 ).normalize();
	scene.add(sunlight);
	
	
	//******************** Arrows to show x, y and z axes in red, green and blue respectively.
	this.arrowX = new THREE.ArrowHelper(new THREE.Vector3( 1, 0, 0 ), origin, 10000, 0xff0000, 1000, 500);
	this.arrowY = new THREE.ArrowHelper(new THREE.Vector3( 0, 1, 0 ), origin, 10000, 0x00ff00, 1000, 500);
	this.arrowZ = new THREE.ArrowHelper(new THREE.Vector3( 0, 0, 1 ), origin, 10000, 0x0000ff, 1000, 500);
	scene.add( this.arrowX );
	scene.add( this.arrowY );
	scene.add( this.arrowZ );			

	var texturePath = "scripts/javascript/texture/lights.jpg";
	var IsTexture = false;
	//******************** Earth model
	add_customGeometry(IsTexture,texturePath);
	//add_buffferGeometry();

	render();

}

function render() 
{
	
	requestAnimationFrame(render);
	//myearth.rotation.x += 0.01;
	//myearth.rotation.y += 0.01;
	//myearth.rotation.z += 0.001;
	renderer.render(scene, camera);
			  
}



function add_buffferGeometry()
{
 	 var buffer_material = new THREE.MeshPhongMaterial( {
 	 			color: 0x999999,
 	 			specular: 0x333333,
 	 			shininess: 50,
 	 			side: THREE.SingleSide,
 	 			vertexColors: THREE.VertexColors
 	 		} );
	 
//	 var buffer_material = new THREE.MeshBasicMaterial( { wireframe: true,color: 0xffff00 } );	
	 var buffer_geometry = create_custom_bufferGeometry(myvertex,myfaces,myflux);
	 var buffer_mesh = new THREE.Mesh( buffer_geometry, buffer_material );
	 buffer_geometry.computeBoundingSphere();
	 scene.add( buffer_mesh );
	 
}



// create the custom BufferGeometry with some attributions
function create_custom_bufferGeometry( verts, faces,flux )
{

	var scale = 1000;	// Convert units from m to km
	var maxflux = Math.max.apply(null,flux);
	var minflux = Math.min.apply(null,flux);
	var fluxlength = maxflux - minflux;
	var colorcoefficient = 255/fluxlength;

	var buffer_geometry = new THREE.BufferGeometry();
	var num_vertex = verts.length;
	var num_face   = faces.length;
	//buffer geometry should have at positions , colors and normals attributes;
	var positions = new Float32Array( num_face * 3 * 3 );
	var normals = new Float32Array( num_face * 3 * 3 );
	var colors = new Float32Array( num_face * 3 * 3 );
	var color = new THREE.Color();

	for( var i = 0 ;i < num_face; i++ )
	{
		var currentColor = (flux[i]-minflux)*colorcoefficient;
		
		color.setRGB( currentColor, 255, 26 ); // set RGB color， red green and blue
			
		for( var j = 0 ; j< 3; j++ )
		{
			positions[i*3+j+0] = verts[faces[i][j]][0]/scale; // x
			positions[i*3+j+1] = verts[faces[i][j]][1]/scale; // y
			positions[i*3+j+2] = verts[faces[i][j]][2]/scale; // z
			
			colors[i*3+j+0] = color.r;
			colors[i*3+j+1] = color.g;
			colors[i*3+j+2] = color.b;

			normals[i*3+j+0] = positions[i*3+j+0]/6371;
			normals[i*3+j+1] = positions[i*3+j+1]/6371;
			normals[i*3+j+2] = positions[i*3+j+2]/6371;

		}
	}

	buffer_geometry.addAttribute('position',new THREE.BufferAttribute(positions,3) );
	buffer_geometry.addAttribute('color',new THREE.BufferAttribute(colors,3) );
	buffer_geometry.addAttribute('normal',new THREE.BufferAttribute(normals,3) );
	
	
	return buffer_geometry;

}



function add_customGeometry(IsTexture, texturePath)
{
	var texture = new THREE.ImageUtils.loadTexture( texturePath );
	
	var material ;
	if(IsTexture == true)
	{
		 material = new THREE.MeshPhongMaterial( { map:texture } );	
	}
	else if(IsTexture == false)
	{
		material = new THREE.MeshBasicMaterial( { wireframe: true,color: 0x000000 } );
	}
	//var material = new THREE.MeshPhongMaterial( { map:texture } );
	//var material = new THREE.MeshBasicMaterial( { wireframe: true,color: 0xffff00 } );
	
	mygeometry = create_custom_sphere(myvertex, myfaces);
	myearth = new THREE.Mesh( mygeometry, material );
	scene.add( myearth );

}


//create custom geometry
function create_custom_sphere(verts, faces)
{
	
	var scale = 1000;	// Convert units from m to km
	
	//var geom = new THREE.Geometry();
	var geom = new THREE.Geometry();

	var num_vertex = verts.length;
	var num_face   = faces.length;
	
	var i;
	
	var u = [];
	var v = [];
	
	var face_u = [0, 0, 0];
	
	for(i=0;i<num_vertex;i++)
	{
		geom.vertices.push( new THREE.Vector3(verts[i][0]/scale,verts[i][1]/scale,verts[i][2]/scale));
		
		u.push( (verts[i][3]-180)/360 );
		v.push( (90+verts[i][4])/180 );
		
		if(u[i]<0){u[i] += 1;}
	}

	for(i=0;i<num_face;i++)
	{
		geom.faces.push( new THREE.Face3(faces[i][0],faces[i][1],faces[i][2]));
		
		face_u[0] = u[ faces[i][0] ];
		face_u[1] = u[ faces[i][1] ];
		face_u[2] = u[ faces[i][2] ];
		
		if(face_u[0]==0 && (face_u[1]>0.75 || face_u[2]>0.75)){face_u[0]=1;}
		if(face_u[1]==0 && (face_u[0]>0.75 || face_u[2]>0.75)){face_u[1]=1;}
		if(face_u[2]==0 && (face_u[0]>0.75 || face_u[1]>0.75)){face_u[2]=1;}
		
		geom.faceVertexUvs[0].push([
			new THREE.Vector2(face_u[0], v[ faces[i][0] ]),
			new THREE.Vector2(face_u[1], v[ faces[i][1] ]),
			new THREE.Vector2(face_u[2], v[ faces[i][2] ])
		]);
		
	}

	geom.computeBoundingSphere();
	geom.computeFaceNormals();
	geom.computeVertexNormals();
	geom.computeLineDistances();
	
	geom.uvsNeedUpdate = true;		// Needed for applying textures correctly
	geom.elementsNeedUpdate = true;	// Needed because faces were changed
	geom.verticesNeedUpdate = true;	// Needed because vertices were changed
	geom.normalsNeedUpdate = true;	// Needed for lighting on faces

	return geom;
	
}


