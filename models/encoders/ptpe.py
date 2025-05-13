import torch
import torch.nn as nn
import numpy as np
import re
from transformers import AutoTokenizer, AutoModel
import math
import os
from utils.fdi_parser import FDIParser

class PolarTextPositionEncoder(nn.Module):
    """
    Polar Text-Position Encoding (PTPE) module
    
    This module extracts anatomical entities from text descriptions,
    maps them to polar coordinates, and generates position encodings.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Load BioClinicalBERT for entity extraction
        self.tokenizer = AutoTokenizer.from_pretrained(config["model"]["text_encoder"]["pretrained"])
        self.text_encoder = AutoModel.from_pretrained(config["model"]["text_encoder"]["pretrained"])
        
        if config["model"]["text_encoder"]["freeze"]:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        
        # Projection layer for position encoding
        self.position_projection = nn.Linear(3, config["model"]["ptpe"]["projection_dim"])
        
        # Initialize FDI parser - try to locate FDI_MATCH.xml
        fdi_xml_path = os.path.join(os.path.dirname(__file__), '../..', 'data', 'FDI_MATCH.xml')
        if not os.path.exists(fdi_xml_path):
            # Try alternate locations
            alt_paths = [
                os.path.join(os.path.dirname(__file__), '../../..', 'FDI_MATCH.xml'),
                os.path.join(os.path.dirname(__file__), '../..', 'FDI_MATCH.xml'),
                'FDI_MATCH.xml'
            ]
            for path in alt_paths:
                if os.path.exists(path):
                    fdi_xml_path = path
                    break
            else:
                print("Warning: FDI_MATCH.xml not found, using default mappings")
                fdi_xml_path = None
                
        self.fdi_parser = FDIParser(fdi_xml_path)
        
        # Pixel size in mm
        self.pixel_size = config["data"]["pixel_size"]
        
    def extract_entities(self, text):
        """
        Extract anatomical entities from text description
        
        Args:
            text (str): Description text in English
            
        Returns:
            dict: Dictionary with extracted entities
        """
        # Use the FDI parser to extract entities
        return self.fdi_parser.extract_entities(text)
    
    def map_to_polar_coordinates(self, entities):
        """
        Map extracted entities to polar coordinates
        
        Args:
            entities (dict): Dictionary with extracted entities
            
        Returns:
            tuple: (r, theta) polar coordinates
        """
        # Default values
        r = 50  # Default radius in pixels
        theta = 0  # Default angle in degrees
        
        # Calculate radius from distance
        if entities["distance"] is not None:
            r = entities["distance"] / self.pixel_size
            
        # Calculate base angle from tooth number
        if entities["tooth_number"] is not None:
            theta = self.fdi_parser.get_angle(entities["tooth_number"])
        # Or use quadrant information
        elif entities["quadrant"] is not None:
            theta = self.fdi_parser.get_quadrant_angle(entities["quadrant"])
            
        # Apply direction adjustment
        if entities["direction"] is not None:
            theta += self.fdi_parser.get_direction_angle(entities["direction"])
            
        # Convert to radians
        theta_rad = math.radians(theta)
        
        return r, theta_rad
    
    def generate_position_encoding(self, r, theta):
        """
        Generate position encoding from polar coordinates
        
        Args:
            r (float): Radius
            theta (float): Angle in radians
            
        Returns:
            torch.Tensor: Position encoding vector [sin(θ), cos(θ), log(r+1)]
        """
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)
        log_r = math.log(r + 1)
        
        ptpe = torch.tensor([sin_theta, cos_theta, log_r])
        return ptpe
    
    def forward(self, text_queries, query_embeddings):
        """
        Process text and generate position-enhanced query embeddings
        
        Args:
            text_queries (list): List of text descriptions
            query_embeddings (torch.Tensor): Query embeddings from DETR [N, num_queries, dim]
            
        Returns:
            torch.Tensor: Position-enhanced query embeddings
        """
        batch_size = query_embeddings.shape[0]
        num_queries = query_embeddings.shape[1]
        device = query_embeddings.device
        
        enhanced_queries = query_embeddings.clone()
        
        for b, text in enumerate(text_queries):
            # Extract entities from text
            entities = self.extract_entities(text)
            
            # Map to polar coordinates
            r, theta = self.map_to_polar_coordinates(entities)
            
            # Generate position encoding
            ptpe = self.generate_position_encoding(r, theta).to(device)
            
            # Project to embedding dimension
            ptpe_embedding = self.position_projection(ptpe.unsqueeze(0))
            
            # Add to all queries
            enhanced_queries[b] = query_embeddings[b] + ptpe_embedding
        
        return enhanced_queries
    
    def convert_polar_to_rect(self, r, theta, image_size):
        """
        Convert polar coordinates to rectangular region
        
        Args:
            r (float): Radius in pixels
            theta (float): Angle in radians
            image_size (tuple): Image size (height, width)
            
        Returns:
            torch.Tensor: [x_min, y_min, x_max, y_max] rectangular region
        """
        # Use image center as origin
        height, width = image_size
        center_x, center_y = width // 2, height // 2
        
        # Calculate center of the lesion
        x_center = center_x + r * math.cos(theta)
        y_center = center_y + r * math.sin(theta)
        
        # Get delta_r and delta_theta from config
        delta_r = self.config["loss"]["delta_r"]
        delta_theta_deg = self.config["loss"]["delta_theta"]
        delta_theta_rad = math.radians(delta_theta_deg)
        
        # Calculate box dimensions
        box_width = 2 * (r * math.sin(delta_theta_rad) + delta_r)
        box_height = box_width  # Make it square for simplicity
        
        # Calculate box corners
        x_min = x_center - box_width / 2
        y_min = y_center - box_height / 2
        x_max = x_center + box_width / 2
        y_max = y_center + box_height / 2
        
        # Ensure box is within image bounds
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(width, x_max)
        y_max = min(height, y_max)
        
        return torch.tensor([x_min, y_min, x_max, y_max]) 
